"""NPZ-native OpenKBP dose evaluation.

Dose score
    Mean absolute voxel error inside the possible-dose mask, first calculated
    per patient and then averaged over patients.

DVH score
    Mean absolute error over the original OpenKBP DVH endpoints:
      * Targets: D1, D95, D99
      * OARs: D0.1cc and mean dose

All dose values are expected in Gy.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


OARS = [
    "Brainstem",
    "SpinalCord",
    "RightParotid",
    "LeftParotid",
    "Esophagus",
    "Larynx",
    "Mandible",
]

TARGETS = ["PTV56", "PTV63", "PTV70"]

STRUCTURES = OARS + TARGETS


def _target_dvh_metrics(dose: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """OpenKBP target endpoints."""
    roi_dose = dose[mask > 0]

    if roi_dose.size == 0:
        raise ValueError("Cannot evaluate DVH metrics for an empty target mask.")

    return {
        "D1": float(np.percentile(roi_dose, 99)),
        "D95": float(np.percentile(roi_dose, 5)),
        "D99": float(np.percentile(roi_dose, 1)),
    }


def _oar_dvh_metrics(
    dose: np.ndarray,
    mask: np.ndarray,
    voxel_dimensions: np.ndarray,
) -> dict[str, float]:
    """OpenKBP OAR endpoints."""
    roi_dose = dose[mask > 0]

    if roi_dose.size == 0:
        raise ValueError("Cannot evaluate DVH metrics for an empty OAR mask.")

    voxel_volume_mm3 = float(np.prod(voxel_dimensions))

    # 0.1 cc = 100 mm^3.
    voxels_in_tenth_cc = max(1.0, float(np.round(100.0 / voxel_volume_mm3)))

    fractional_volume_to_evaluate = (
        100.0 - voxels_in_tenth_cc / roi_dose.size * 100.0
    )

    # Preserve the original metric definition rather than silently clipping.
    if not 0.0 <= fractional_volume_to_evaluate <= 100.0:
        raise ValueError(
            "OAR is smaller than the volume required for D0.1cc evaluation."
        )

    return {
        "D_0.1_cc": float(
            np.percentile(roi_dose, fractional_volume_to_evaluate)
        ),
        "mean": float(np.mean(roi_dose)),
    }


def _structure_is_present(
    case: np.lib.npyio.NpzFile,
    structure_name: str,
    structure_index: int,
) -> bool:
    """Mirror the original 'structure file exists' logic for dense NPZ masks."""
    if "structure_present" in case.files:
        present = np.asarray(case["structure_present"]).astype(bool)

        if structure_index >= len(present):
            raise ValueError("structure_present has an unexpected length.")

        return bool(present[structure_index])

    return bool(np.any(case[structure_name] > 0))


def _load_prediction(prediction_path: Path) -> np.ndarray:
    if not prediction_path.exists():
        raise FileNotFoundError(f"Missing prediction: {prediction_path}")

    with np.load(prediction_path, allow_pickle=False) as prediction_case:
        if "dose" not in prediction_case.files:
            raise KeyError(
                f"{prediction_path} must contain a dense array called 'dose'."
            )

        dose = np.asarray(
            prediction_case["dose"],
            dtype=np.float32,
        )

    if not np.isfinite(dose).all():
        raise ValueError(f"Prediction contains NaN or inf: {prediction_path}")

    return dose


class DoseEvaluator:
    """Evaluate dense NPZ predictions against reference patient NPZ files."""

    def __init__(
        self,
        reference_patient_paths: Iterable[Path | str],
        prediction_dir: Path | str,
    ):
        self.reference_patient_paths = [
            Path(path)
            for path in reference_patient_paths
        ]
        self.prediction_dir = Path(prediction_dir)

        self.patient_dose_scores: list[float] = []
        self.dvh_errors: list[float] = []

        self.dose_score: float | None = None
        self.dvh_score: float | None = None

    def evaluate(self) -> None:
        self.patient_dose_scores = []
        self.dvh_errors = []

        if not self.reference_patient_paths:
            raise ValueError("No reference patients were supplied.")

        for patient_path in self.reference_patient_paths:
            patient_id = patient_path.name
            reference_path = patient_path / "patient.npz"
            prediction_path = (
                self.prediction_dir
                / f"{patient_id}.npz"
            )

            if not reference_path.exists():
                raise FileNotFoundError(
                    f"Missing reference NPZ: {reference_path}"
                )

            prediction = _load_prediction(prediction_path)

            with np.load(reference_path, allow_pickle=False) as case:
                required = {
                    "dose",
                    "possible_dose_mask",
                    "voxel_dimensions",
                    *STRUCTURES,
                }

                missing = required.difference(case.files)

                if missing:
                    raise KeyError(
                        f"{reference_path} is missing: {sorted(missing)}"
                    )

                reference = np.asarray(
                    case["dose"],
                    dtype=np.float32,
                )
                possible_dose_mask = np.asarray(
                    case["possible_dose_mask"]
                )
                voxel_dimensions = np.asarray(
                    case["voxel_dimensions"],
                    dtype=np.float32,
                )

                if prediction.shape != reference.shape:
                    raise ValueError(
                        f"{patient_id}: prediction shape {prediction.shape} "
                        f"does not match reference {reference.shape}"
                    )

                feasible = possible_dose_mask > 0

                if not np.any(feasible):
                    raise ValueError(
                        f"{patient_id}: possible-dose mask is empty"
                    )

                patient_dose_score = float(
                    np.mean(
                        np.abs(
                            prediction[feasible]
                            - reference[feasible]
                        )
                    )
                )

                self.patient_dose_scores.append(
                    patient_dose_score
                )

                for structure_index, structure_name in enumerate(STRUCTURES):
                    if not _structure_is_present(
                        case,
                        structure_name,
                        structure_index,
                    ):
                        continue

                    structure_mask = np.asarray(
                        case[structure_name]
                    )

                    if structure_name in TARGETS:
                        reference_metrics = _target_dvh_metrics(
                            reference,
                            structure_mask,
                        )
                        prediction_metrics = _target_dvh_metrics(
                            prediction,
                            structure_mask,
                        )
                    else:
                        reference_metrics = _oar_dvh_metrics(
                            reference,
                            structure_mask,
                            voxel_dimensions,
                        )
                        prediction_metrics = _oar_dvh_metrics(
                            prediction,
                            structure_mask,
                            voxel_dimensions,
                        )

                    for metric_name in reference_metrics:
                        self.dvh_errors.append(
                            abs(
                                reference_metrics[metric_name]
                                - prediction_metrics[metric_name]
                            )
                        )

        if not self.dvh_errors:
            raise ValueError("No DVH endpoints were available for evaluation.")

        self.dose_score = float(
            np.mean(self.patient_dose_scores)
        )
        self.dvh_score = float(
            np.mean(self.dvh_errors)
        )

    def get_scores(self) -> tuple[float, float]:
        if self.dose_score is None or self.dvh_score is None:
            raise RuntimeError(
                "Call evaluate() before get_scores()."
            )

        return self.dose_score, self.dvh_score
