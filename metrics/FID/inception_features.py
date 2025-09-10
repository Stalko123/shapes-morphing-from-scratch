import io
import os
import math
import pickle
import urllib.request
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def _load_pkl_from_url_or_path(path_or_url: str):
	"""
	Lightweight alternative to NVIDIA's misc.load_pkl that supports local paths or HTTP(S) URLs.
	"""
	if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
		with urllib.request.urlopen(path_or_url) as f:
			data = f.read()
			return pickle.load(io.BytesIO(data))
	else:
		with open(path_or_url, "rb") as f:
			return pickle.load(f)


class InceptionFeatureExtractor:
	"""
	Compute InceptionV3 features using NVIDIA's pre-trained pickled network
	("inception_v3_features.pkl").

	Inputs are expected as PyTorch tensors in [B, C, H, W] format. Values can be
	in [-1, 1] (common for diffusion models) or [0, 1]. They will be internally
	converted to uint8 [0, 255] as expected by NVIDIA's feature extractor.
	"""

	def __init__(
		self,
		inception_pkl: str = "https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/inception_v3_features.pkl",
		device: Optional[torch.device] = None,
	):
		self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
		self.inception: nn.Module = _load_pkl_from_url_or_path(inception_pkl)
		self.inception.eval().to(self.device)

	@staticmethod
	def _to_uint8_nchw(x: torch.Tensor) -> torch.Tensor:
		"""Convert tensor in [-1,1] or [0,1] to uint8 [0,255] NCHW. If grayscale, tile to 3 channels."""
		assert x.ndim == 4 and x.size(1) in (1, 3), "Input must be [B, C, H, W] with C in {1,3}."
		if x.dtype != torch.float32:
			x = x.float()
		# Heuristic normalization detection
		min_val = float(x.min().item())
		max_val = float(x.max().item())
		if min_val >= -1.01 and max_val <= 1.01:
			x = (x.clamp(-1, 1) * 0.5 + 0.5)  # [-1,1] -> [0,1]
		else:
			x = x.clamp(0, 1)
		# If grayscale, tile to 3 channels for Inception
		if x.size(1) == 1:
			x = x.repeat(1, 3, 1, 1)
		x = (x * 255.0).round().clamp(0, 255).to(torch.uint8)
		return x

	@torch.no_grad()
	def compute_activations(self, images: torch.Tensor, batch_size: int = 64) -> np.ndarray:
		"""
		Compute 2048-D features for a batch tensor or an iterable of batches.

		Args:
			images: Either a tensor [N, C, H, W] or an iterable yielding tensors [B, C, H, W].
			batch_size: If `images` is a single tensor, we split it into mini-batches of this size.

		Returns:
			np.ndarray of shape [N, 2048]
		"""
		features_list = []
		if isinstance(images, torch.Tensor):
			N = images.size(0)
			for start in range(0, N, batch_size):
				end = min(start + batch_size, N)
				batch = images[start:end].to(self.device)
				uint8_batch = self._to_uint8_nchw(batch)
				feats = self.inception(uint8_batch, return_features=True)
				features_list.append(feats.cpu().numpy())
		else:
			for batch in images:
				batch = batch[0] if isinstance(batch, (tuple, list)) else batch
				batch = batch.to(self.device)
				uint8_batch = self._to_uint8_nchw(batch)
				feats = self.inception(uint8_batch, return_features=True)
				features_list.append(feats.cpu().numpy())
		return np.concatenate(features_list, axis=0)

	@staticmethod
	def compute_stats(activations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		"""Return (mu, sigma) for activations [N, D]."""
		mu = np.mean(activations, axis=0)
		sigma = np.cov(activations, rowvar=False)
		return mu, sigma 