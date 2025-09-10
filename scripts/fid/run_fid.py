import argparse

from metrics.FID.fid import FIDCalculator
from utils.parsing.inference.argparser_inference import p as inference_parser


def build_cli_parser():
	parser = argparse.ArgumentParser(description="Compute FID for a trained DDPM model")
	# Reuse inference args (model, dataset_name, version, path_to_weights, path_to_yaml, ...)
	# We'll parse them via the existing parser to build the model.
	parser.add_argument("--num_gen", type=int, default=5000, help="Number of generated samples")
	parser.add_argument("--ref_split", type=str, default="test", choices=["train", "test"], help="Reference split")
	parser.add_argument("--ref_max", type=int, default=None, help="Max items from reference dataset (default: all)")
	parser.add_argument("--batch_size", type=int, default=64)
	parser.add_argument("--inception_pkl", type=str, default="https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/inception_v3_features.pkl")
	return parser


def main():
	# Parse inference args first (so we can reuse existing config and weights)
	# We clone the existing parser to avoid consuming args prematurely
	inference_args = inference_parser.parse_known_args()[0]
	# Our local FID args
	fid_args = build_cli_parser().parse_args()

	calc = FIDCalculator(inception_pkl=fid_args.inception_pkl)
	ref_stats = calc.compute_dataset_stats(
		dataset_name=inference_args.dataset_name,
		batch_size=fid_args.batch_size,
		split=fid_args.ref_split,
		max_items=fid_args.ref_max,
	)
	gen_stats = calc.generate_samples_features(
		parsed_args=inference_args,
		num_samples=fid_args.num_gen,
		batch_size=fid_args.batch_size,
	)
	fid = calc.compute_fid(ref_stats, gen_stats)
	print(f"FID: {fid:.4f}")


if __name__ == "__main__":
	main() 