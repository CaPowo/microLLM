import argparse
from pathlib import Path

import torch

from data import read_file
from model import Model
from tokenizer import Tokenizer


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "tiny_shakespeare.txt"
DEFAULT_CHECKPOINT_PATH = BASE_DIR / "best_model.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Load best checkpoint and generate text.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--start", type=str, default="\n")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    return parser.parse_args()


def build_model_from_checkpoint(checkpoint: dict) -> Model:
    model = Model(
        vocab_size=checkpoint["vocab_size"],
        n_embd=checkpoint["n_embd"],
        block_size=checkpoint["block_size"],
        num_heads=checkpoint["num_heads"],
        n_layer=checkpoint["n_layer"],
        dropout=checkpoint.get("dropout", 0.0),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def main():
    args = parse_args()

    if args.temperature <= 0:
        raise ValueError("temperature 必须大于 0。")

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"找不到 checkpoint: {args.checkpoint}\n"
            "请先运行: uv run python microLLM/train.py"
        )

    text = read_file(str(args.data))
    tok = Tokenizer(text)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint["vocab_size"] != tok.vocab_size:
        raise ValueError(
            f"checkpoint vocab_size={checkpoint['vocab_size']}，"
            f"但当前 tokenizer vocab_size={tok.vocab_size}。请确认训练和生成使用同一份数据。"
        )

    try:
        start_ids = tok.encode(args.start)
    except KeyError as exc:
        raise ValueError(f"起始文本里有训练词表不存在的字符: {exc}") from exc

    model = build_model_from_checkpoint(checkpoint)
    context = torch.tensor([start_ids], dtype=torch.long)
    out = model.generate(
        context,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    step = checkpoint.get("step", "unknown")
    best_val_loss = checkpoint.get("best_val_loss", "unknown")
    print(f"loaded best checkpoint from step {step}: val loss {best_val_loss}")
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
