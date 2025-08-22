#!/usr/bin/env python
"""
Minimal text-only test script for Gemma3n using Liger-Kernel patches.

- Loads a Gemma3n model from Hugging Face
- Optionally applies Liger's Gemma3n text patch (RMSNorm, RoPE, GeGLU, fused CE)
- Runs a simple generation from a text prompt

Usage examples:

  python examples/huggingface/test_gemma3n_text.py \
      --model-id RTannous/gemma-3n-E2B-it \
      --with-liger --prompt "Explain transformers in 2 sentences."

  HF_TOKEN=hf_xxx python examples/huggingface/test_gemma3n_text.py \
      --model-id RTannous/gemma-3n-E2B-it --with-liger

Notes:
- This patches only the text stack. Vision/audio paths are untouched.
- Requires a transformers version that includes the Gemma3n classes.
"""

import argparse
import os
import sys
import time

import torch


def main():
    parser = argparse.ArgumentParser(description="Test Gemma3n text-only with Liger kernels")
    parser.add_argument("--model-id", type=str, default="RTannous/gemma-3n-E2B-it", help="HF model repo or path")
    parser.add_argument("--hf-token", type=str, default=None, help="HF token (or use env HF_TOKEN)")
    parser.add_argument("--with-liger", action="store_true", help="Apply Liger Gemma3n text patch")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Write a short poem about models that run fast.",
        help="Text prompt",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max new tokens to generate")
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype to load the model",
    )
    parser.add_argument("--device", type=str, default=None, help="torch device (default: auto)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Resolve device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Resolve dtype
    if args.dtype == "auto":
        if device == "cuda":
            # Prefer bfloat16 on modern GPUs
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
    else:
        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[args.dtype]

    # HF token
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    # Import here so we can show clearer errors if transformers lacks Gemma3n
    try:
        from transformers import AutoProcessor, AutoModelForCausalLM
        from transformers import set_seed
        # Ensure Gemma3n module exists in this transformers version
        import transformers.models.gemma3n  # noqa: F401
    except Exception as e:
        print("[Error] transformers does not provide Gemma3n in this environment:", file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(1)

    set_seed(args.seed)

    if args.with_liger:
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_gemma3n_text

            # Patch global classes before model load for best coverage
            apply_liger_kernel_to_gemma3n_text(
                rope=True,
                cross_entropy=False,
                fused_linear_cross_entropy=False,  # safer for Gemma3n config differences
                rms_norm=True,
                geglu=True,
            )
            print("[Info] Applied Liger Gemma3n text patch (RoPE, RMSNorm, GeGLU, fused CE)")
        except Exception as e:
            print("[Error] Failed to apply Liger Gemma3n text patch:", file=sys.stderr)
            print(e, file=sys.stderr)
            sys.exit(1)

    print(f"[Info] Loading model: {args.model_id}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        attn_implementation="eager",  # Gemma3 recommends eager for training; safe for testing too
        token=hf_token,
    )
    model.to(device)
    # If Liger patching is requested, patch the loaded instance to ensure MLP forward binding
    if args.with_liger:
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_gemma3n_text

            apply_liger_kernel_to_gemma3n_text(
                rope=True,
                cross_entropy=False,
                fused_linear_cross_entropy=False,
                rms_norm=True,
                geglu=True,
                model=model,
            )
            print("[Info] Patched Gemma3n model instance with Liger (text-only)")
        except Exception as e:
            print("[Warn] Failed to patch Gemma3n instance:", e, file=sys.stderr)
    load_s = time.time() - t0
    print(f"[Info] Model loaded in {load_s:.2f}s on {device} with dtype {dtype}.")

    print("[Info] Loading processor")
    processor = AutoProcessor.from_pretrained(args.model_id, token=hf_token)

    print(f"[Info] Prompt: {args.prompt!r}")
    inputs = processor(text=args.prompt, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    print("[Info] Running generation ... (first run may include compile/warmup)")
    # Warmup forward to trigger any lazy init/compilation
    with torch.inference_mode():
        _ = model(**{k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask")})

    gen_t0 = time.time()
    with torch.inference_mode():
        generated_ids = model.generate(
            **{k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask")},
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            top_p=0.95,
            top_k=64,
        )
    gen_s = time.time() - gen_t0
    print(f"[Info] Generation finished in {gen_s:.2f}s")

    # Try to decode via processor if available, else fallback to tokenizer
    try:
        text_out = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    except Exception:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(args.model_id, token=hf_token)
            text_out = tok.batch_decode(generated_ids, skip_special_tokens=True)
        except Exception as e:
            print("[Warn] Failed to decode outputs:", e, file=sys.stderr)
            text_out = [str(generated_ids)]

    print("\n=== Output ===")
    for i, s in enumerate(text_out):
        print(f"[{i}] {s}")


if __name__ == "__main__":
    main()
