"""Convert original OpenKBP patient CSV folders to one compressed NPZ per patient.

The original CSV files are deliberately left untouched so the original OpenKBP
metric/evaluation code can still be used. 

This file was coded by an LLM. Use its content with care. 

Example
-------
python prepare_npz.py --data-root provided-data
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

GRID_SHAPE = (128, 128, 128)
GRID_SIZE = int(np.prod(GRID_SHAPE))

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


def _read_csv(path: Path):
    """Read one file in the sparse OpenKBP CSV representation."""
    if path.stem == "voxel_dimensions":
        return np.loadtxt(path, dtype=np.float32)

    frame = pd.read_csv(path, index_col=0)

    # OpenKBP masks are stored as indices with an empty/NaN data column.
    if frame.shape[1] == 0 or frame.isnull().values.any():
        return np.asarray(frame.index, dtype=np.int64)

    return {
        "indices": np.asarray(frame.index, dtype=np.int64),
        "data": np.asarray(frame.iloc[:, 0].values, dtype=np.float32),
    }


def _dense_values(path: Path, dtype=np.float32) -> np.ndarray:
    sparse = _read_csv(path)
    out = np.zeros(GRID_SIZE, dtype=dtype)
    out[sparse["indices"]] = sparse["data"].astype(dtype, copy=False)
    return out.reshape(GRID_SHAPE)


def _dense_mask(path: Path) -> np.ndarray:
    out = np.zeros(GRID_SIZE, dtype=np.uint8)
    out[_read_csv(path)] = 1
    return out.reshape(GRID_SHAPE)


def convert_patient(patient_dir: Path, overwrite: bool = False) -> Path:
    output_path = patient_dir / "patient.npz"
    if output_path.exists() and not overwrite:
        return output_path

    ct_path = patient_dir / "ct.csv"
    mask_path = patient_dir / "possible_dose_mask.csv"
    voxel_path = patient_dir / "voxel_dimensions.csv"

    if not ct_path.exists() or not mask_path.exists():
        raise FileNotFoundError(f"{patient_dir} is not a complete OpenKBP patient folder")

    arrays: dict[str, np.ndarray] = {
        "ct": _dense_values(ct_path, np.float32),
        "possible_dose_mask": _dense_mask(mask_path),
        "voxel_dimensions": _read_csv(voxel_path).astype(np.float32),
    }

    dose_path = patient_dir / "dose.csv"
    if dose_path.exists():
        arrays["dose"] = _dense_values(dose_path, np.float32)

    present = []
    for structure in STRUCTURES:
        structure_path = patient_dir / f"{structure}.csv"
        is_present = structure_path.exists()
        present.append(is_present)
        arrays[structure] = (
            _dense_mask(structure_path)
            if is_present
            else np.zeros(GRID_SHAPE, dtype=np.uint8)
        )

    arrays["structure_present"] = np.asarray(present, dtype=np.uint8)
    np.savez_compressed(output_path, **arrays)
    return output_path


def patient_directories(data_root: Path, splits: list[str] | None = None) -> list[Path]:
    split_dirs = []
    if splits:
        split_dirs = [data_root / split for split in splits]
    else:
        split_dirs = sorted(p for p in data_root.glob("*-pats") if p.is_dir())

    patient_dirs: list[Path] = []
    for split_dir in split_dirs:
        patient_dirs.extend(sorted(p for p in split_dir.glob("pt_*") if p.is_dir()))
    return patient_dirs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("provided-data"))
    parser.add_argument("--split", action="append", dest="splits", help="e.g. train-pats")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    patients = patient_directories(args.data_root, args.splits)
    if not patients:
        raise RuntimeError(f"No patient folders found below {args.data_root}")

    for patient_dir in tqdm(patients, desc="Converting OpenKBP patients"):
        convert_patient(patient_dir, overwrite=args.overwrite)

    print(f"Prepared {len(patients)} patient NPZ files.")


if __name__ == "__main__":
    main()

