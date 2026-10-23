"""MONAI/PyTorch input pipeline for OpenKBP patient data pre-converted to npz."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from monai.data import DataLoader, Dataset, list_data_collate
from monai.transforms import (
    Compose, ConcatItemsd, DeleteItemsd, EnsureTyped,
    MapTransform, RandCropByPosNegLabeld, ScaleIntensityRanged,
)

OARS = ["Brainstem", "SpinalCord", "RightParotid", "LeftParotid",
        "Esophagus", "Larynx", "Mandible"]
TARGETS = ["PTV56", "PTV63", "PTV70"]
STRUCTURES = OARS + TARGETS
INPUT_KEYS = ["ct", *STRUCTURES, "possible_dose_mask"]   # 12 channels
N_INPUT_CHANNELS = len(INPUT_KEYS)

# dose is kept in Gy in the arrays that reach the writer; the network only ever
# sees/optimises dose / DOSE_SCALE.
DOSE_SCALE = 100.0


class LoadOpenKBPd(MapTransform):
    """Load the openKBP challenge data converted to NPZ."""

    def __init__(self, keys=("npz_path",), include_dose=True):
        super().__init__(keys)
        self.include_dose = include_dose

    def __call__(self, data):
        d = dict(data)
        path = Path(d["npz_path"])
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist. Check the supplied course data.")

        with np.load(path, allow_pickle=False) as case:
            for key in INPUT_KEYS:
                d[key] = case[key][None, ...]                 # channel-first
            if self.include_dose:
                if "dose" not in case.files:
                    raise KeyError(f"Training case {path} does not contain dose")
                d["dose"] = case["dose"][None, ...] / DOSE_SCALE
            d["voxel_dimensions"] = case["voxel_dimensions"].astype(np.float32)
            if "structure_present" in case.files:
                d["structure_present"] = case["structure_present"].astype(np.uint8)
        return d


class NPZDataSet:
    def __init__(self, training_paths=(), validation_paths=(), cache_root="npz_cache",
                 patch_size: Sequence[int] = (64, 64, 64),
                 train_batch_size=2, val_batch_size=1, num_workers=8,
                 dose_scale=DOSE_SCALE):
        global DOSE_SCALE
        DOSE_SCALE = dose_scale

        self.train_records = self._case_records(training_paths)
        self.val_records = self._case_records(validation_paths)
        self.cache_root, self.patch_size = Path(cache_root), tuple(patch_size)
        self.train_batch_size, self.val_batch_size = train_batch_size, val_batch_size
        self.num_workers = num_workers

    @staticmethod
    def _case_records(patient_paths):
        records = []
        for p in [Path(x) for x in patient_paths]:
            records.append({
                "patient_id": p.name,                      
                "patient_path": str(p),
                "npz_path": str(p / "patient.npz"),
            })
        return records


    # ---- transforms ------------------------------------------------------ #
    # THIS IS A GOOD PLACE TO EXPLORE 
    # ---------------------------------------------------------------------- #
    def _deterministic_transforms(self, include_dose=True):
        tensor_keys = INPUT_KEYS + (["dose"] if include_dose else [])
        return [
            LoadOpenKBPd(include_dose=include_dose),
            # cast before concatenating: mixed float32/uint8 promotes to float64
            ScaleIntensityRanged(keys=["ct"], 
                                 a_min=-1024,
                                 a_max=3071,
                                 b_min=-1.0, 
                                 b_max=1.0, 
                                 clip=True),
            EnsureTyped(keys=tensor_keys, dtype=torch.float32, track_meta=False), 
            ConcatItemsd(keys=INPUT_KEYS, 
                         name="image", 
                         dim=0),
            # keep the feasible-dose mask separately for the masked loss
            DeleteItemsd(keys=["ct", *STRUCTURES]),
        ]

    def _random_training_transforms(self, patch_size):
        # last transform in the chain: returns list[dict]
        return [
            RandCropByPosNegLabeld(
                keys=["image", "dose", "possible_dose_mask"],
                label_key="possible_dose_mask",
                image_key="image",               # fallback if the mask is empty
                spatial_size=tuple(patch_size),
                pos=1, neg=0, num_samples=1,
                allow_smaller=False,
            )
        ]

    def train_transform(self, no_random: bool = False) -> Compose:
        steps = self._deterministic_transforms(include_dose=True)
        if not no_random:
            steps+= self._random_training_transforms(self.patch_size)
        return Compose(steps)

    def val_transform(self, include_dose=True) -> Compose:
        return Compose(self._deterministic_transforms(include_dose=include_dose))

    # ---- datasets -------------------------------------------------------- #

    def train_dataset(self, no_random: bool = False):
        return Dataset(data=self.train_records, transform=self.train_transform(no_random))
        
    def val_dataset(self, include_dose=True):
        return Dataset(data=self.val_records, transform=self.val_transform(include_dose))

    # ---- loaders --------------------------------------------------------- #
    def _loader(self, ds, batch_size, shuffle):
        kwargs = dict(batch_size=batch_size, shuffle=shuffle, num_workers=self.num_workers,
                      collate_fn=list_data_collate, pin_memory=torch.cuda.is_available())
        if self.num_workers > 0:
            kwargs.update(persistent_workers=True, prefetch_factor=2)
        return DataLoader(ds, **kwargs)

    def train_dataloader(self, no_random=False):
        return self._loader(self.train_dataset(no_random), self.train_batch_size, True)

    def val_dataloader(self):
        return self._loader(self.val_dataset(include_dose=True), self.val_batch_size, False)

    def predict_dataloader(self):
        return self._loader(self.val_dataset(include_dose=False), 1, False)