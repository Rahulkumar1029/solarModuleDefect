from __future__ import annotations

from typing import Any

import logging
import torch
from lightning import LightningModule

log = logging.getLogger(__name__)


class YOLOv11Module(LightningModule):
    """A thin Lightning wrapper for an Ultralytics YOLO model.

    Notes:
    - This wrapper exposes the underlying Ultralytics `YOLO` model for inference.
    - Training with PyTorch Lightning is not implemented by default because Ultralytics
      provides a mature training loop. Use the `ultralytics` CLI or call
      `from ultralytics import YOLO; YOLO('yolov8n.pt').train(data=..., epochs=...)`.
    - This module provides `configure_optimizers` so it can be extended with a
      custom `training_step` implementation if you prefer Lightning training.
    """

    def __init__(self, model_name: str = "yolov8n.pt", lr: float = 1e-3) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        try:
            from ultralytics import YOLO  # type: ignore

            self._ultralytics_available = True
        except Exception:  # pragma: no cover - optional dependency
            YOLO = None  # type: ignore
            self._ultralytics_available = False

        if not self._ultralytics_available:
            log.warning("Ultralytics package not available. Install `ultralytics` to use this module.")
            self.model = torch.nn.Identity()
        else:
            # load Ultralytics YOLO; underlying .model is a torch.nn.Module
            self._ul_model = YOLO(model_name)  # pyright/typing: dynamic
            # underlying PyTorch nn.Module
            self.model = getattr(self._ul_model, "model", None) or torch.nn.Identity()

    def forward(self, imgs: Any) -> Any:
        """Run inference through the underlying model.

        `imgs` can be a tensor batch or a list of image arrays depending on the
        model internals. This method simply forwards to the underlying `model`.
        """
        return self.model(imgs)

    def training_step(self, batch, batch_idx):
        raise RuntimeError(
            "training_step is not implemented in this wrapper.\n"
            "For training, either: (1) use the `ultralytics` training API:"
            " `from ultralytics import YOLO; YOLO('yolov8n.pt').train(data='src/data/files/data.yaml', epochs=50)`\n"
            "or (2) implement a custom `training_step` here that computes losses from model outputs."
        )

    def configure_optimizers(self):
        # provide a sensible default optimizer so users can extend this Module
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            return None
        opt = torch.optim.Adam(params, lr=self.hparams.lr)
        return opt
