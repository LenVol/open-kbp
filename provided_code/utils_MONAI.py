import os
from pathlib import Path
from typing import Optional
import shutil

import numpy as np
import matplotlib.pyplot as plt


def get_paths(directory_path: Path, extension: Optional[str] = None) -> list[Path]:
    """
    Get the paths of every file contained in `directory_path` that also has the extension `extension` if one is provided.
    """
    all_paths = []

    if not directory_path.is_dir():
        pass
    elif extension is None:
        dir_list = os.listdir(directory_path)
        for name in dir_list:
            if "." != name[0]:  # Ignore hidden files
                all_paths.append(directory_path / str(name))
    else:
        data_root = Path(directory_path)
        for file_path in data_root.glob("*.{}".format(extension)):
            file_path = Path(file_path)
            if "." != file_path.stem[0]:
                all_paths.append(file_path)

    return all_paths

def plot_validation_case(patient_path, prediction_dir):
    """Plot CT, reference dose, prediction, and absolute error for one patient."""
    patient_path = Path(patient_path)
    prediction_dir = Path(prediction_dir)

    reference_path = patient_path / "patient.npz"
    prediction_path = prediction_dir / f"{patient_path.name}.npz"

    with np.load(reference_path, allow_pickle=False) as case:
        ct = case["ct"]
        reference_dose = case["dose"]

        # Use the slice containing the largest total target area.
        target = (
            case["PTV56"]
            + case["PTV63"]
            + case["PTV70"]
        )

    with np.load(prediction_path, allow_pickle=False) as prediction_case:
        predicted_dose = prediction_case["dose"]

    if reference_dose.shape != predicted_dose.shape:
        raise ValueError(
            f"Reference shape {reference_dose.shape} does not match "
            f"prediction shape {predicted_dose.shape}"
        )

    target_area_per_slice = target.sum(axis=(0, 1))

    if target_area_per_slice.max() > 0:
        slice_index = int(np.argmax(target_area_per_slice))
    else:
        # Fallback for an unusual case without a target.
        slice_index = reference_dose.shape[2] // 2

    absolute_error = np.abs(predicted_dose - reference_dose)

    # Use the same dose scale for reference and prediction.
    dose_max = max(
        float(reference_dose.max()),
        float(predicted_dose.max()),
    )

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4),
        constrained_layout=True,
    )

    axes[0].imshow(
        ct[:, :, slice_index],
        cmap="gray",
    )
    axes[0].set_title("CT")

    reference_image = axes[1].imshow(
        reference_dose[:, :, slice_index],
        vmin=0,
        vmax=dose_max,
        cmap="inferno",
    )
    axes[1].set_title("Reference dose")

    axes[2].imshow(
        predicted_dose[:, :, slice_index],
        vmin=0,
        vmax=dose_max,
        cmap="inferno",
    )
    axes[2].set_title("Predicted dose")

    error_image = axes[3].imshow(
        absolute_error[:, :, slice_index],
        vmin=0,
        cmap="magma",
    )
    axes[3].set_title("Absolute error")

    for ax in axes:
        ax.axis("off")

    fig.colorbar(
        reference_image,
        ax=axes[1:3],
        label="Dose [Gy]",
        shrink=0.8,
    )
    fig.colorbar(
        error_image,
        ax=axes[3],
        label="|Error| [Gy]",
        shrink=0.8,
    )

    fig.suptitle(
        f"{patient_path.name} — slice {slice_index}"
    )

    plt.show()

def create_submission_zip(
    prediction_dir: Path,
    submission_zip: Path,
) -> None:
    """Create a flat submission.zip containing one NPZ file per patient."""
    prediction_dir = Path(prediction_dir)
    submission_zip = Path(submission_zip)

    prediction_files = sorted(prediction_dir.glob("*.npz"))

    if not prediction_files:
        raise FileNotFoundError(
            f"No prediction NPZ files found in {prediction_dir}"
        )

    submission_zip.parent.mkdir(parents=True, exist_ok=True)

    if submission_zip.exists():
        submission_zip.unlink()

    created_zip = Path(
        shutil.make_archive(
            base_name=str(submission_zip.with_suffix("")),
            format="zip",
            root_dir=prediction_dir,
        )
    )

    if created_zip != submission_zip:
        raise RuntimeError(
            f"Unexpected submission archive path: {created_zip}"
        )

    print()
    print(f"Submission created: {submission_zip}")
    print(f"Prediction files: {len(prediction_files)}")
    print(
        "Submit this ZIP unchanged. It contains one <patient_id>.npz "
        "prediction file per test patient."
    )
