"""
PyTorch Dataset and DataLoader for Face Anti-Spoofing
Author: Team STT 12 - Face Anti-Spoofing
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Callable
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image

from .texture import LBPFeatureExtractor, get_lbp_extractor
from .preprocessing import get_image_transforms


class FaceAntiSpoofingDataset(Dataset):
    """
    Multimodal Dataset returning:
    - rgb_tensor: [3, 224, 224] normalized image for ResNet34 backbone
    - lbp_tensor: [lbp_dim] normalized feature vector for Texture stream
    - label: 0 for LIVE, 1 for SPOOF
    - metadata: dictionary containing video and subject metadata
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        transform: Optional[Callable] = None,
        lbp_extractor: Optional[LBPFeatureExtractor] = None,
        cache_lbp: bool = True
    ):
        self.df = metadata_df.reset_index(drop=True)
        self.transform = transform
        self.lbp_extractor = lbp_extractor or LBPFeatureExtractor()
        self.cache_lbp = cache_lbp
        self._lbp_cache: Dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        row = self.df.iloc[idx]
        img_path = str(row["image_path"])

        # Load RGB image
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at {img_path}")

        # Open image using PIL for torchvision transforms consistency
        pil_img = Image.open(img_path).convert("RGB")
        np_img = np.array(pil_img)

        # 1. Texture (LBP) feature vector
        if self.cache_lbp and idx in self._lbp_cache:
            lbp_tensor = self._lbp_cache[idx]
        else:
            lbp_feat = self.lbp_extractor.extract_features(np_img)
            lbp_tensor = torch.tensor(lbp_feat, dtype=torch.float32)
            if self.cache_lbp:
                self._lbp_cache[idx] = lbp_tensor

        # 2. RGB image transformation
        if self.transform:
            rgb_tensor = self.transform(np_img)
        else:
            rgb_tensor = transforms.ToTensor()(pil_img)

        # 3. Label: 0 = LIVE, 1 = SPOOF
        label = int(row["label"])
        label_tensor = torch.tensor(label, dtype=torch.long)

        # 4. Metadata dictionary
        metadata = {
            "image_path": img_path,
            "video_name": str(row.get("video_name", "")),
            "subject_id": str(row.get("subject_id", "")),
            "label_name": str(row.get("label_name", "LIVE" if label == 0 else "SPOOF")),
            "device": str(row.get("device", "")),
            "attack_type": str(row.get("attack_type", "none")),
            "frame_idx": int(row.get("frame_idx", 0)),
            "split": str(row.get("split", ""))
        }

        return rgb_tensor, lbp_tensor, label_tensor, metadata


def compute_class_weights(df: pd.DataFrame) -> torch.Tensor:
    """
    Computes inverse frequency class weights to balance CrossEntropyLoss.
    Live (0) is minority (approx 25%), Spoof (1) is majority (approx 75%).
    """
    labels = df["label"].values
    num_live = np.sum(labels == 0)
    num_spoof = np.sum(labels == 1)
    total = len(labels)

    if num_live == 0 or num_spoof == 0:
        return torch.tensor([1.0, 1.0], dtype=torch.float32)

    weight_live = total / (2.0 * num_live)
    weight_spoof = total / (2.0 * num_spoof)
    return torch.tensor([weight_live, weight_spoof], dtype=torch.float32)


def create_dataloaders(
    metadata_csv_path: str,
    config: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Builds Train, Validation, and Test DataLoaders from metadata CSV.
    returns: (train_loader, val_loader, test_loader, class_weights)
    """
    df = pd.read_csv(metadata_csv_path)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    image_size = config.get("dataset", {}).get("image_size", 224)
    batch_size = config.get("training", {}).get("batch_size", 16)
    num_workers = config.get("training", {}).get("num_workers", 0)

    train_transform = get_image_transforms(image_size=image_size, is_train=True)
    eval_transform = get_image_transforms(image_size=image_size, is_train=False)

    lbp_extractor = get_lbp_extractor(config)

    train_dataset = FaceAntiSpoofingDataset(
        train_df, transform=train_transform, lbp_extractor=lbp_extractor, cache_lbp=True
    )
    val_dataset = FaceAntiSpoofingDataset(
        val_df, transform=eval_transform, lbp_extractor=lbp_extractor, cache_lbp=True
    )
    test_dataset = FaceAntiSpoofingDataset(
        test_df, transform=eval_transform, lbp_extractor=lbp_extractor, cache_lbp=True
    )

    class_weights = compute_class_weights(train_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader, class_weights
