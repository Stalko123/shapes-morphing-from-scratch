import math
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .inception_features import InceptionFeatureExtractor
from utils.parsing.inference.args_inference import InferenceArgs
from DDPM.ddpm import DDPM
from loaders.dataloader import Loader


def _frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
	"""Compute Frechet distance between two Gaussians.

	Tries to use SciPy's sqrtm for numerical stability if available. Falls back
	to eigen decomposition otherwise.
	"""
	diff = mu1 - mu2
	try:
		from scipy.linalg import sqrtm  # type: ignore
		cov_prod_sqrt = sqrtm(sigma1.dot(sigma2))
		if np.iscomplexobj(cov_prod_sqrt):
			cov_prod_sqrt = cov_prod_sqrt.real
		fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * cov_prod_sqrt)
		return float(fid)
	except Exception:
		vals, vecs = np.linalg.eigh(sigma1 @ sigma2)
		sqrt_prod = (vecs * np.sqrt(np.clip(vals, a_min=0.0, a_max=None))) @ vecs.T
		fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * sqrt_prod)
		return float(np.real(fid))


class FIDCalculator:
	"""
	High-level FID computation utility.

	- Computes reference stats (mu, sigma) on a dataset via `Loader`.
	- Generates samples via your DDPM `generate_one_sample` using `InferenceArgs`.
	- Computes FID between the two sets.
	"""

	def __init__(
		self,
		inception_pkl: str = "https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/inception_v3_features.pkl",
		device: Optional[torch.device] = None,
	):
		self.extractor = InceptionFeatureExtractor(inception_pkl=inception_pkl, device=device)
		self.device = self.extractor.device

	def compute_dataset_stats(
		self,
		dataset_name: str = "CIFAR10",
		batch_size: int = 128,
		num_workers: int = 4,
		split: str = "test",
		max_items: Optional[int] = None,
	) -> Tuple[np.ndarray, np.ndarray]:
		ldr = Loader(name=dataset_name, batch_size=batch_size, num_workers=num_workers, split=split).dataloader
		seen = 0
		def limited_batches():
			nonlocal seen
			for batch in ldr:
				images = batch[0] if isinstance(batch, (tuple, list)) else batch
				if max_items is None:
					yield images
					continue
				remaining = max_items - seen
				if remaining <= 0:
					break
				if images.size(0) <= remaining:
					seen += images.size(0)
					yield images
				else:
					# yield only what is needed to reach max_items
					seen += remaining
					yield images[:remaining]
					break
		# Pass iterable so the extractor's iterable path is exercised
		acts = self.extractor.compute_activations(limited_batches())
		return self.extractor.compute_stats(acts)

	def _build_ddpm(self, parsed_args) -> DDPM:
		inf_args = InferenceArgs(parsed_args)
		model = inf_args.model.to(self.device)
		model.eval()
		ddpm = DDPM(inf_args)
		ddpm.denoiser = model
		return ddpm

	@torch.no_grad()
	def generate_samples_features(
		self,
		parsed_args,
		num_samples: int,
		batch_size: int = 32,
	) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Generate `num_samples` images using the model defined by `parsed_args` (Inference argparser)
		and return (mu, sigma) of Inception activations.
		"""
		ddpm = self._build_ddpm(parsed_args)
		images_list = []
		for _ in range(num_samples):
			frames_or_img = ddpm.generate_one_sample(return_intermediates=False, device=self.device)
			img = frames_or_img
			if isinstance(img, list):
				img = img[-1]
			images_list.append(img.cpu())
		images = torch.stack(images_list, dim=0)
		acts = self.extractor.compute_activations(images, batch_size=batch_size)
		return self.extractor.compute_stats(acts)

	def compute_fid(
		self,
		ref_stats: Tuple[np.ndarray, np.ndarray],
		gen_stats: Tuple[np.ndarray, np.ndarray],
	) -> float:
		mu1, sigma1 = ref_stats
		mu2, sigma2 = gen_stats
		return _frechet_distance(mu1, sigma1, mu2, sigma2) 