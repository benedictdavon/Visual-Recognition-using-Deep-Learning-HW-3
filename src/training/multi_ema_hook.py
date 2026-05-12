from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from mmengine.dist import is_main_process
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.registry import HOOKS, MODELS


@HOOKS.register_module()
class MultiEMAHook(Hook):
    """Maintain and save multiple EMA shadows during one training run.

    Each EMA is saved as a normal inference-loadable MMDetection checkpoint so
    the user can evaluate one EMA alone or ensemble several EMA checkpoints
    after training.
    """

    priority = "NORMAL"

    def __init__(
        self,
        momentums: list[float],
        names: list[str] | None = None,
        *,
        ema_type: str = "ExponentialMovingAverage",
        interval: int = 1,
        begin_iter: int = 0,
        update_buffers: bool = False,
        device: str | None = None,
        save_deploy: bool = True,
        save_interval: int = 1,
        save_begin: int = 1,
        save_last: bool = True,
        out_dir: str = "ema_checkpoints",
        save_in_main_checkpoint: bool = False,
        strict_load: bool = False,
    ) -> None:
        if not momentums:
            raise ValueError("MultiEMAHook requires at least one EMA momentum.")
        if any(float(momentum) <= 0.0 or float(momentum) >= 1.0 for momentum in momentums):
            raise ValueError(f"EMA momentums must be in (0, 1), got {momentums!r}.")
        if names is None:
            names = [f"ema{index}" for index in range(len(momentums))]
        if len(names) != len(momentums):
            raise ValueError("MultiEMAHook names and momentums must have the same length.")
        if len(set(names)) != len(names):
            raise ValueError(f"MultiEMAHook names must be unique, got {names!r}.")
        if interval <= 0:
            raise ValueError(f"interval must be positive, got {interval!r}.")
        if save_interval <= 0:
            raise ValueError(f"save_interval must be positive, got {save_interval!r}.")
        if save_begin <= 0:
            raise ValueError(f"save_begin must be positive, got {save_begin!r}.")

        self.momentums = [float(momentum) for momentum in momentums]
        self.names = [str(name) for name in names]
        self.ema_type = str(ema_type)
        self.interval = int(interval)
        self.begin_iter = int(begin_iter)
        self.update_buffers = bool(update_buffers)
        self.device = device
        self.save_deploy = bool(save_deploy)
        self.save_interval = int(save_interval)
        self.save_begin = int(save_begin)
        self.save_last = bool(save_last)
        self.out_dir = str(out_dir)
        self.save_in_main_checkpoint = bool(save_in_main_checkpoint)
        self.strict_load = bool(strict_load)
        self.ema_models: dict[str, Any] = {}

    def before_run(self, runner: Any) -> None:
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        self.src_model = model
        self.ema_models = {}
        for name, momentum in zip(self.names, self.momentums):
            self.ema_models[name] = MODELS.build(
                {
                    "type": self.ema_type,
                    "momentum": momentum,
                    "interval": self.interval,
                    "device": self.device,
                    "update_buffers": self.update_buffers,
                },
                default_args={"model": self.src_model},
            )

    def after_train_iter(
        self,
        runner: Any,
        batch_idx: int,
        data_batch: Any = None,
        outputs: dict[str, Any] | None = None,
    ) -> None:
        if runner.iter + 1 < self.begin_iter:
            for ema_model in self.ema_models.values():
                self._copy_source_to_ema(ema_model)
            return
        for ema_model in self.ema_models.values():
            ema_model.update_parameters(self.src_model)

    def after_train_epoch(self, runner: Any) -> None:
        if not self.save_deploy:
            return
        epoch = int(runner.epoch) + 1
        should_save = epoch >= self.save_begin and epoch % self.save_interval == 0
        is_last = self.save_last and epoch >= int(runner.max_epochs)
        if should_save or is_last:
            self._save_deploy_checkpoints(runner, epoch)

    def before_save_checkpoint(self, runner: Any, checkpoint: dict[str, Any]) -> None:
        if not self.save_in_main_checkpoint:
            return
        checkpoint["multi_ema_state_dict"] = {
            name: ema_model.state_dict() for name, ema_model in self.ema_models.items()
        }
        checkpoint["multi_ema_meta"] = self._meta(runner)

    def after_load_checkpoint(self, runner: Any, checkpoint: dict[str, Any]) -> None:
        state_dicts = checkpoint.get("multi_ema_state_dict")
        if state_dicts:
            for name, ema_model in self.ema_models.items():
                if name in state_dicts:
                    ema_model.load_state_dict(state_dicts[name], strict=self.strict_load)
            return
        for ema_model in self.ema_models.values():
            self._copy_source_to_ema(ema_model)

    def _copy_source_to_ema(self, ema_model: Any) -> None:
        src_state = self.src_model.state_dict()
        ema_state = ema_model.module.state_dict()
        for key, value in ema_state.items():
            value.data.copy_(src_state[key].data.to(value.device))

    def _meta(self, runner: Any) -> dict[str, Any]:
        return {
            "epoch": int(runner.epoch) + 1,
            "iter": int(runner.iter),
            "ema_names": self.names,
            "ema_momentums": self.momentums,
            "ema_type": self.ema_type,
            "interval": self.interval,
            "update_buffers": self.update_buffers,
        }

    def _state_dict_for_deploy(self, ema_model: Any) -> dict[str, torch.Tensor]:
        state_dict = ema_model.module.state_dict()
        return {key: value.detach().cpu() for key, value in state_dict.items()}

    def _save_deploy_checkpoints(self, runner: Any, epoch: int) -> None:
        if not is_main_process():
            return
        out_dir = Path(runner.work_dir) / self.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, Any] = {
            "epoch": epoch,
            "iter": int(runner.iter),
            "checkpoints": {},
        }
        base_meta = self._meta(runner)
        for name, momentum in zip(self.names, self.momentums):
            ema_model = self.ema_models[name]
            payload = {
                "meta": {
                    **base_meta,
                    "ema_name": name,
                    "ema_momentum": momentum,
                    "source": "MultiEMAHook deploy checkpoint",
                },
                "state_dict": self._state_dict_for_deploy(ema_model),
            }
            checkpoint_path = out_dir / f"epoch_{epoch}_{name}.pth"
            latest_path = out_dir / f"latest_{name}.pth"
            torch.save(payload, checkpoint_path)
            torch.save(payload, latest_path)
            manifest["checkpoints"][name] = {
                "momentum": momentum,
                "path": str(checkpoint_path),
                "latest_path": str(latest_path),
            }

        manifest_path = out_dir / f"epoch_{epoch}_multi_ema_manifest.json"
        latest_manifest_path = out_dir / "latest_multi_ema_manifest.json"
        manifest_text = json.dumps(manifest, indent=2, sort_keys=True)
        manifest_path.write_text(manifest_text + "\n", encoding="utf-8")
        latest_manifest_path.write_text(manifest_text + "\n", encoding="utf-8")
