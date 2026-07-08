# 负责训练数据减小 loss
# 核心思路：取一批数据 → 前向算 loss → 反向算梯度 → 朝下坡挪一步 → 重复几千次

import argparse
import os
from pathlib import Path

import torch

from data import read_file, make_data_tensor, train_val_split, get_batch
from model import Model
from tokenizer import build_tokenizer


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "tiny_shakespeare.txt"
DEFAULT_CHECKPOINT_PATH = BASE_DIR / "best_model.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a tiny decoder-only Transformer.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
    parser.add_argument("--bpe-merges", type=int, default=200)
    parser.add_argument("--bpe-min-frequency", type=int, default=2)
    parser.add_argument("--bpe-train-chars", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-steps", type=int, default=20000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--start", type=str, default="")
    parser.add_argument("--num-threads", type=int, default=0)
    return parser.parse_args()


def validate_data_size(train, val, block_size: int, batch_size: int):
    min_len = block_size + batch_size
    if len(train) < min_len:
        raise ValueError(
            f"训练集太短：len(train)={len(train)}，至少需要 block_size + batch_size = {min_len}。"
        )
    if len(val) < min_len:
        raise ValueError(
            f"验证集太短：len(val)={len(val)}，至少需要 block_size + batch_size = {min_len}。"
        )


def make_start_ids(tok, start: str):
    if start:
        return tok.encode(start)
    return [0]


def configure_cpu_threads(num_threads: int):
    if num_threads <= 0:
        num_threads = os.cpu_count() or 1

    interop_threads = max(1, min(4, num_threads))
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(interop_threads)
    print(
        f"PyTorch CPU threads: intra_op={torch.get_num_threads()}, "
        f"interop={torch.get_num_interop_threads()}",
        flush=True,
    )


def main():
    args = parse_args()
    configure_cpu_threads(args.num_threads)

    if args.temperature <= 0:
        raise ValueError("temperature 必须大于 0。")
    if args.n_embd % args.num_heads != 0:
        raise ValueError("n_embd 必须能被 num_heads 整除。")
    if not args.data.exists():
        raise FileNotFoundError(f"找不到训练数据: {args.data}")

    # 1. 加载训练数据
    print(f"读取训练数据: {args.data}", flush=True)
    text = read_file(str(args.data))
    print(f"文本长度: {len(text)} 字符", flush=True)
    print(f"构建 tokenizer: {args.tokenizer}", flush=True)
    tokenizer_text = text
    if args.tokenizer == "bpe" and args.bpe_train_chars > 0 and len(text) > args.bpe_train_chars:
        tokenizer_text = text[:args.bpe_train_chars]
        print(
            f"BPE 仅使用前 {len(tokenizer_text)} 字符学习 merge 规则；"
            "词表仍覆盖完整训练文本。",
            flush=True,
        )
    tok = build_tokenizer(
        tokenizer_text,
        tokenizer_type=args.tokenizer,
        bpe_merges=args.bpe_merges,
        bpe_min_frequency=args.bpe_min_frequency,
        vocab_text=text,
    )
    print("编码训练数据为 token id...", flush=True)
    if hasattr(tok, "encode_with_progress"):
        data = torch.tensor(tok.encode_with_progress(text), dtype=torch.long)
    else:
        data = make_data_tensor(text, tok)
    print(f"token 数量: {len(data)}", flush=True)
    train, val = train_val_split(data)
    validate_data_size(train, val, args.block_size, args.batch_size)
    print(f"train token 数量: {len(train)}, val token 数量: {len(val)}", flush=True)

    # 2. 创建模型
    print("创建模型，开始训练...", flush=True)
    model = Model(
        n_embd=args.n_embd,
        block_size=args.block_size,
        vocab_size=tok.vocab_size,
        num_heads=args.num_heads,
        n_layer=args.n_layer,
        dropout=args.dropout,
    )

    # 3. 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split, data_src in [("train", train), ("val", val)]:
            losses = torch.zeros(args.eval_iters)
            for k in range(args.eval_iters):
                xb, yb = get_batch(data_src, args.block_size, args.batch_size)
                _, loss = model(xb, yb)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    best_val_loss = float("inf")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    # 4. 循环训练
    for i in range(args.num_steps):
        xb, yb = get_batch(train, args.block_size, args.batch_size)
        logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if i % args.eval_interval == 0:
            losses = estimate_loss()
            print(
                f"step {i}: train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "tokenizer": tok.to_config(),
                        "chars": getattr(tok, "chars", None),
                        "vocab_size": tok.vocab_size,
                        "n_embd": args.n_embd,
                        "block_size": args.block_size,
                        "num_heads": args.num_heads,
                        "n_layer": args.n_layer,
                        "dropout": args.dropout,
                        "learning_rate": args.learning_rate,
                        "bpe_train_chars": args.bpe_train_chars,
                        "best_val_loss": best_val_loss,
                        "step": i,
                        "data_path": str(args.data),
                    },
                    args.checkpoint,
                )
                print(f"  saved best checkpoint: val loss {best_val_loss:.4f}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(
        f"loaded best checkpoint from step {checkpoint['step']}: "
        f"val loss {checkpoint['best_val_loss']:.4f}"
    )

    start_ids = make_start_ids(tok, args.start)
    start = torch.tensor([start_ids], dtype=torch.long)
    out = model.generate(
        start,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
