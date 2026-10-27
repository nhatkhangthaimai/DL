"""
Local Binary Patterns (LBP) Texture Feature Extraction
Author: Team STT 12 - Face Anti-Spoofing
"""

import numpy as np
from typing import Tuple, Optional


class LBPFeatureExtractor:
    """
    Extracts multi-block Local Binary Patterns (LBP) texture descriptors from face crops.
    Captures micro-texture artifacts, print rasters, screen moiré patterns, and specular reflections.
    """

    def __init__(
        self,
        radius: int = 1,
        n_points: int = 8,
        method: str = "uniform",
        grid_x: int = 4,
        grid_y: int = 4,
    ):
        self.radius = radius
        self.n_points = n_points
        self.method = method
        self.grid_x = grid_x
        self.grid_y = grid_y

        # Precompute uniform pattern lookup table for P=8
        self._lookup_table = self._build_uniform_lookup_table(self.n_points)
        # Uniform patterns for P=8 has 59 bins: 58 uniform patterns + 1 non-uniform bin
        # If method == "uniform_simple": 10 bins (P+2)
        if self.method == "uniform_10":
            self.num_bins = self.n_points + 2  # 10 bins
        else:
            self.num_bins = 59  # Standard Ojala uniform LBP bins for P=8

        self.feature_dim = self.num_bins * self.grid_x * self.grid_y

    @staticmethod
    def _count_transitions(val: int, n_bits: int = 8) -> int:
        """Count 0->1 and 1->0 transitions in circular bit sequence."""
        bits = [(val >> i) & 1 for i in range(n_bits)]
        transitions = 0
        for i in range(n_bits):
            if bits[i] != bits[(i + 1) % n_bits]:
                transitions += 1
        return transitions

    def _build_uniform_lookup_table(self, n_bits: int = 8) -> np.ndarray:
        """Builds lookup table mapping 0..255 to uniform pattern bin indices."""
        lut = np.zeros(256, dtype=np.int32)
        uniform_idx = 0
        non_uniform_idx = 58  # 59th bin (index 58) for non-uniform patterns

        for val in range(256):
            if self._count_transitions(val, n_bits) <= 2:
                lut[val] = uniform_idx
                uniform_idx += 1
            else:
                lut[val] = non_uniform_idx
        return lut

    def compute_lbp_image(self, gray: np.ndarray) -> np.ndarray:
        """
        Fast vectorized computation of LBP image (radius=1, n_points=8).
        gray: 2D uint8 or float32 image [H, W]
        returns: 2D int32 image of uniform pattern indices [H-2, W-2]
        """
        if gray.ndim == 3:
            # Convert BGR/RGB to gray if 3-channel
            gray = 0.2989 * gray[:, :, 0] + 0.5870 * gray[:, :, 1] + 0.1140 * gray[:, :, 2]
        gray = gray.astype(np.float32)

        H, W = gray.shape
        if H < 3 or W < 3:
            return np.zeros((H, W), dtype=np.int32)

        center = gray[1:-1, 1:-1]
        lbp_code = np.zeros_like(center, dtype=np.int32)

        # 8 neighbors clockwise starting from top-left:
        # (-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)
        neighbors = [
            gray[0:-2, 0:-2],  # (-1, -1) -> bit 0
            gray[0:-2, 1:-1],  # (-1,  0) -> bit 1
            gray[0:-2, 2:],    # (-1, +1) -> bit 2
            gray[1:-1, 2:],    # ( 0, +1) -> bit 3
            gray[2:,   2:],    # (+1, +1) -> bit 4
            gray[2:,   1:-1],  # (+1,  0) -> bit 5
            gray[2:,   0:-2],  # (+1, -1) -> bit 6
            gray[1:-1, 0:-2],  # ( 0, -1) -> bit 7
        ]

        for p, neighbor in enumerate(neighbors):
            lbp_code |= ((neighbor >= center).astype(np.int32) << p)

        # Map to uniform bins
        uniform_lbp = self._lookup_table[lbp_code]
        return uniform_lbp

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extracts spatially enhanced LBP histogram features from input image.
        image: RGB or Grayscale image [H, W, 3] or [H, W]
        returns: 1D normalized float32 feature vector of shape [feature_dim]
        """
        lbp_img = self.compute_lbp_image(image)
        h, w = lbp_img.shape

        block_h = h // self.grid_y
        block_w = w // self.grid_x

        histograms = []
        for gy in range(self.grid_y):
            for gx in range(self.grid_x):
                y1 = gy * block_h
                y2 = (gy + 1) * block_h if gy < self.grid_y - 1 else h
                x1 = gx * block_w
                x2 = (gx + 1) * block_w if gx < self.grid_x - 1 else w

                sub_block = lbp_img[y1:y2, x1:x2]
                hist, _ = np.histogram(sub_block, bins=self.num_bins, range=(0, self.num_bins))

                # Normalize block histogram (L1 norm with epsilon)
                hist_sum = hist.sum()
                if hist_sum > 0:
                    hist = hist.astype(np.float32) / (hist_sum + 1e-7)
                else:
                    hist = np.zeros(self.num_bins, dtype=np.float32)

                histograms.append(hist)

        # Concatenate block histograms into global texture descriptor
        feature_vector = np.concatenate(histograms)

        # L2 normalization over entire feature vector
        norm = np.linalg.norm(feature_vector)
        if norm > 1e-7:
            feature_vector = feature_vector / norm

        return feature_vector.astype(np.float32)


def get_lbp_extractor(config: Optional[dict] = None) -> LBPFeatureExtractor:
    """Factory function to build LBPFeatureExtractor from config dictionary."""
    if config and "lbp" in config:
        cfg = config["lbp"]
        return LBPFeatureExtractor(
            radius=cfg.get("radius", 1),
            n_points=cfg.get("n_points", 8),
            method=cfg.get("method", "uniform"),
            grid_x=cfg.get("grid_x", 4),
            grid_y=cfg.get("grid_y", 4),
        )
    return LBPFeatureExtractor(radius=1, n_points=8, method="uniform", grid_x=4, grid_y=4)
