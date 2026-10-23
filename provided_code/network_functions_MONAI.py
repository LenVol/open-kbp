"""Training and inference for the MONAI/PyTorch OpenKBP course baseline.

The API stays close to the orignal openKBP challenge. 
However, the inner workings were changed from keras/tensorflow to MONAI/torch

"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from tqdm.auto import tqdm

from .monai_data import DOSE_SCALE, NPZDataSet
from .network_architectures_MONAI import DoseInference


def masked_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    possible_dose_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean absolute dose error inside the feasible-dose region."""
    mask = possible_dose_mask.float()
    error = torch.abs(prediction - target)
    return (error * mask).sum() / mask.sum().clamp_min(1.0)


def save_dose_npz(
    dose: np.ndarray,
    output_path: Path,
) -> None:
    """Save one dense dose prediction in Gy."""
    dose = np.asarray(
        dose,
        dtype=np.float32,
    )

    if not np.isfinite(dose).all():
        raise ValueError(
            "Refusing to save a prediction containing NaN or inf."
        )

    np.savez_compressed(
        output_path,
        dose=dose,
    )


class PredictionModel:
    """Train and apply the MONAI/PyTorch dose-prediction baseline."""

    def __init__(
        self,
        data_loader,
        results_dir: Path,
        model_name: str = "baseline",
        stage: str = "train",
        learning_rate: float = 1e-3,
    ):
        self.data_loader = data_loader
        self.results_dir = Path(results_dir)
        self.model_name = model_name
        self.stage = stage

        self.model_root = (
            self.results_dir
            / self.model_name
        )
        self.model_dir = (
            self.model_root
            / "models"
        )
        self.prediction_dir = (
            self.model_root
            / f"{stage}-predictions"
        )

        self.model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        if stage != "train":
            self.prediction_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # ================================================================
        # CHALLENGE EXPERIMENT AREA: training/inference hyperparameters
        # ================================================================
        self.learning_rate = learning_rate

        self.model = DoseInference().to(
            self.device
        )

        # ================================================================
        # CHALLENGE EXPERIMENT AREA: optimizer
        # ================================================================
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        self.use_amp = (
            self.device.type == "cuda"
        )

        self.scaler = torch.amp.GradScaler('cuda',
            enabled=self.use_amp
        )

    def _checkpoint_path(
        self,
        epoch: int,
    ) -> Path:
        return (
            self.model_dir
            / f"epoch_{epoch}.pt"
        )

    def _save_checkpoint(
        self,
        epoch: int,
    ) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            self._checkpoint_path(epoch),
        )

    def _load_checkpoint(
        self,
        epoch: int,
    ) -> None:
        path = self._checkpoint_path(
            epoch
        )

        if not path.exists():
            raise FileNotFoundError(
                f"No checkpoint found at {path}"
            )

        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    def _trim_model_history(
        self,
        keep_model_history: int,
    ) -> None:
        checkpoints = sorted(
            self.model_dir.glob(
                "epoch_*.pt"
            ),
            key=lambda path: int(
                path.stem.split("_")[-1]
            ),
        )

        for old_checkpoint in checkpoints[
            :-keep_model_history
        ]:
            old_checkpoint.unlink(
                missing_ok=True
            )

    def train_model(
        self,
        num_epochs: int,
        save_frequency: int = 1,
        keep_model_history: int = 20,
    ) -> None:

        data_loader = self.data_loader

        
        for epoch in range(
            1,
            num_epochs + 1,
        ):
            self.model.train()

            running_loss = 0.0
            n_batches = 0


            #progress bar
            progress = tqdm(
                data_loader,
                desc=(
                    f"Epoch {epoch}/{num_epochs}"
                ),
                leave=False,
            )

            for batch in progress:
                images = batch["image"].float().to(
                    self.device,
                    non_blocking=True,
                )
                target = batch["dose"].float().to(
                    self.device,
                    non_blocking=True,
                )
                mask = batch[
                    "possible_dose_mask"
                ].float().to(
                    self.device,
                    non_blocking=True,
                )

                self.optimizer.zero_grad(
                    set_to_none=True
                )

                with torch.amp.autocast('cuda',
                    enabled=self.use_amp
                ):
                    prediction = self.model(
                        images
                    )

                    # ====================================================
                    # CHALLENGE EXPERIMENT AREA: loss
                    # ====================================================
                    loss = masked_mae(
                        prediction,
                        target,
                        mask,
                    )

                self.scaler.scale(
                    loss
                ).backward()

                self.scaler.step(
                    self.optimizer
                )

                self.scaler.update()

                running_loss += float(
                    loss.detach().cpu()
                )
                n_batches += 1

                progress.set_postfix(
                    masked_mae_gy=(
                        f"{DOSE_SCALE * running_loss / n_batches:.3f}"
                    )
                )

            epoch_loss = (
                running_loss
                / max(1, n_batches)
            )

            print(
                f"Epoch {epoch:03d}: "
                f"masked MAE = "
                f"{epoch_loss * DOSE_SCALE:.3f} Gy"
            )

            if (
                epoch % save_frequency == 0
                or epoch == num_epochs
            ):
                self._save_checkpoint(
                    epoch
                )
                # self._trim_model_history(
                #     keep_model_history
                # )


    def predict_dose(
        self,
        epoch: int,
        batch_size: int = 1,
        roi_size: tuple[int] = (64,64,64)
    ) -> None:
        self._load_checkpoint(
            epoch
        )
        self.model.eval()

        progress = tqdm(
                self.data_loader,
                desc=f"Predicting {self.stage}",
            )

        with torch.no_grad():
            for batch in progress:
                images = batch["image"].float().to(
                    self.device,
                    non_blocking=True,
                )

                mask = batch[
                    "possible_dose_mask"
                ].float().to(
                    self.device,
                    non_blocking=True,
                )

                with torch.amp.autocast('cuda',
                    enabled=self.use_amp
                ):
                    prediction = (
                        sliding_window_inference(
                            inputs=images,
                            roi_size=roi_size,
                            sw_batch_size=batch_size,
                            predictor=self.model,

                            # ===========================================
                            # CHALLENGE EXPERIMENT AREA:
                            # inference settings
                            # ===========================================
                            overlap=0.25,
                        )
                    )

                # Network target is dose / DOSE_SCALE.
                # Convert predictions back to Gy.
                prediction = (
                    prediction
                    * DOSE_SCALE
                )

                # Match OpenKBP feasible-dose convention.
                prediction = (
                    prediction
                    * mask
                )

                patient_id = (
                    batch["patient_id"][0]
                )

                output_path = (
                    self.prediction_dir
                    / f"{patient_id}.npz"
                )

                save_dose_npz(
                    prediction[
                        0,
                        0,
                    ].detach().cpu().numpy(),
                    output_path,
                )
