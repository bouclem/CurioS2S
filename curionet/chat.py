import os
import sys
import torch

from .model import CurioNet
from .data import (
    VOCAB_SIZE,
    PAD_IDX,
    BOS_IDX,
    EOS_IDX,
    encode,
    decode,
    IDX_TO_CHAR,
)

CHECKPOINT_PATH = os.path.join("checkpoints", "curionet_best.pt")


def load_model(checkpoint_path: str = CHECKPOINT_PATH, device: str = "cpu") -> CurioNet:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. Run training first: python -m curionet.train"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = CurioNet(
        src_vocab_size=config["src_vocab_size"],
        tgt_vocab_size=config["tgt_vocab_size"],
        dim=config["dim"],
        num_layers=config["num_layers"],
        num_heads=config.get("num_heads", 4),
        dropout=config["dropout"],
        padding_idx=config["padding_idx"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def ask(model: CurioNet, question: str, device: str = "cpu", max_len: int = 30, min_len: int = 10) -> str:
    src = torch.tensor([encode(question)], dtype=torch.long).to(device)
    output = model.generate(src, max_len=max_len, bos_idx=BOS_IDX, eos_idx=EOS_IDX, min_len=min_len)
    return decode(output[0].cpu().tolist())


def ask_stream(model: CurioNet, question: str, device: str = "cpu", max_len: int = 30, min_len: int = 10):
    model.eval()
    src = torch.tensor([encode(question)], dtype=torch.long).to(device)
    enc_out = model.encoder(src)

    tgt = torch.full((1, 1), BOS_IDX, dtype=torch.long, device=device)

    for step in range(max_len - 1):
        logits = model.decoder(tgt, enc_out)
        top2 = logits[:, -1, :].topk(2, dim=-1)
        next_token = top2.indices[:, 0].item()

        # Don't allow eos until min_len is reached — use second-best token
        if step < min_len and next_token == EOS_IDX:
            next_token = top2.indices[:, 1].item()

        if next_token == EOS_IDX:
            break
        if next_token == PAD_IDX:
            continue

        char = IDX_TO_CHAR.get(next_token, "?")
        yield char

        tgt = torch.cat([tgt, torch.tensor([[next_token]], dtype=torch.long, device=device)], dim=1)


def chat():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = load_model(device=device)
    print("CurioNet loaded! Type text to continue. (streaming output)")
    print("Type 'quit' or 'exit' to leave.\n")

    while True:
        try:
            prompt = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if prompt.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        if not prompt:
            continue

        sys.stdout.write("CurioNet> ")
        sys.stdout.flush()
        for char in ask_stream(model, prompt, device=device, max_len=60):
            sys.stdout.write(char)
            sys.stdout.flush()
        sys.stdout.write("\n")
        print()


if __name__ == "__main__":
    chat()
