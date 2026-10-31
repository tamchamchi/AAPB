"""
AAPB inference script for rare concept generation on the RareBench dataset.

Usage:
    python src/inference.py \
        --test_dir ../benchmark/rewrite_gepa \
        --prompt_keys r2f_prompt r2f_prompt_rewrite \
        --out_path output/rewrite_gepa --device cuda:0
"""

import os
import json
import argparse

import torch

from pipelines.pipeline_aapb_sd3 import AAPBDiffusion3Pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="AAPB inference for RareBench")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--test_file",
        type=str,
        help="Path to R2F prompt file (JSON format)",
    )
    source.add_argument(
        "--test_dir",
        type=str,
        help="Directory of RareBench JSON files",
    )
    parser.add_argument(
        "--prompt_keys",
        nargs="+",
        choices=("r2f_prompt", "r2f_prompt_rewrite"),
        default=["r2f_prompt"],
        help="Prompt versions to generate (default: r2f_prompt)",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="output/",
        help="Base output directory",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.0,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--gamma_t",
        type=float,
        default=None,
        help="Fixed blending coefficient (0.0-1.0). If None, uses adaptive AAPB",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="stabilityai/stable-diffusion-3-medium",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="refs/pr/26",
        help="Model revision",
    )
    parser.add_argument(
        "--device",
        type=str
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.test_dir:
        test_files = sorted(
            os.path.join(args.test_dir, name)
            for name in os.listdir(args.test_dir)
            if name.endswith(".json")
        )
        if not test_files:
            raise ValueError(f"No JSON files found in {args.test_dir}")
    else:
        test_files = [args.test_file]

    # Load pipeline
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    torch_dtype = (
        torch.float16
        if device.startswith("cuda")
        else torch.float32
    )
    pipe = AAPBDiffusion3Pipeline.from_pretrained(
        args.model_id,
        revision=args.revision,
        torch_dtype=torch_dtype,
    )
    pipe = pipe.to(device)

    for test_file in test_files:
        test_case = os.path.splitext(os.path.basename(test_file))[0]
        for suffix in ("_gpt4", "_human", "_llama3"):
            test_case = test_case.removesuffix(suffix)

        with open(test_file, "r") as f:
            prompts = json.load(f)
        print(f"Loaded {len(prompts)} prompts from {test_file}")

        for prompt_key in args.prompt_keys:
            # Keep the original single-file output layout for existing commands.
            if args.test_file and args.prompt_keys == ["r2f_prompt"]:
                save_path = os.path.join(args.out_path, test_case)
            else:
                save_path = os.path.join(args.out_path, prompt_key, test_case)
            os.makedirs(save_path, exist_ok=True)

            for i, (prompt, prompt_data) in enumerate(prompts.items()):
                if prompt_key not in prompt_data:
                    raise ValueError(f"Missing {prompt_key} for {prompt!r} in {test_file}")
                out_file = os.path.join(
                    save_path, f"{i}_seed{args.seed}_{prompt.rstrip()}.png"
                )
                if os.path.exists(out_file):
                    print(f"Skipping {prompt_key}: {prompt} (already exists)")
                    continue

                print(f"[{test_case}/{prompt_key}/{i}] {prompt}")
                image = pipe(
                    r2f_prompts={"r2f_prompt": prompt_data[prompt_key]},
                    batch_size=1,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    gamma_t=args.gamma_t,
                    seed=args.seed,
                ).images[0]

                image.save(out_file)
                print(f"  Saved -> {out_file}")


if __name__ == "__main__":
    main()
