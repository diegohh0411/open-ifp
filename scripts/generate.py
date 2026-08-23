#!/usr/bin/env python3
"""Load a Hugging Face causal LM through mlx-lm (Metal), not PyTorch."""

from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download
from mlx_lm import generate, load

DEFAULT_MODEL = "Qwen/Qwen2-0.5B-Instruct"
DEFAULT_MODEL_REVISION = "c540970f9e29518b1d8f06ab8b24cba66ad77b6d"
DEFAULT_PROMPT = "In one sentence, what is instruction-following pruning?"


def main() -> None:
    parser = argparse.ArgumentParser(description="mlx-lm generate from a Hugging Face model id")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face repo id (default: Qwen/Qwen2-0.5B-Instruct)",
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_MODEL_REVISION,
        help=f"immutable Hugging Face revision (default: {DEFAULT_MODEL_REVISION})",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    snapshot_path = snapshot_download(
        repo_id=args.model,
        revision=args.revision,
        local_files_only=True,
    )
    print(f"loading {args.model}@{args.revision} from {snapshot_path} via mlx-lm…")
    model, tokenizer = load(snapshot_path)
    messages = [{"role": "user", "content": args.prompt}]
    if tokenizer.chat_template is not None:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = args.prompt
    text = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens)
    print(text)


if __name__ == "__main__":
    main()
