import os
import sys
import torch

from .model import CurioS2S
from .data import (
    VOCAB_SIZE,
    PAD_IDX,
    BOS_IDX,
    EOS_IDX,
    encode,
    decode,
    load_samples,
    IDX_TO_CHAR,
)

CHECKPOINT_PATH = os.path.join("checkpoints", "best.pt")


def load_model(checkpoint_path: str = CHECKPOINT_PATH, device: str = "cpu") -> CurioS2S:
    """Load a trained CurioS2S model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. Run training first: python -m curios2s.train"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = CurioS2S(
        src_vocab_size=config["src_vocab_size"],
        tgt_vocab_size=config["tgt_vocab_size"],
        dim=config["dim"],
        num_layers=config["num_layers"],
        kernel_size=config["kernel_size"],
        dropout=config["dropout"],
        padding_idx=config["padding_idx"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def ask(model: CurioS2S, question: str, device: str = "cpu", max_len: int = 30) -> str:
    """Ask the model a question and return its answer."""
    src = torch.tensor([encode(question)], dtype=torch.long).to(device)
    output = model.generate(src, max_len=max_len, bos_idx=BOS_IDX, eos_idx=EOS_IDX)
    return decode(output[0].cpu().tolist())


def ask_stream(model: CurioS2S, question: str, device: str = "cpu", max_len: int = 30):
    """Ask the model a question, yielding one character at a time.

    Generator that produces each character as it's generated.
    """
    model.eval()
    src = torch.tensor([encode(question)], dtype=torch.long).to(device)
    enc_out = model.encoder(src)

    tgt = torch.full((1, 1), BOS_IDX, dtype=torch.long, device=device)
    finished = False

    for _ in range(max_len - 1):
        logits = model.decoder(tgt, enc_out)
        next_token = logits[:, -1, :].argmax(dim=-1).item()

        if next_token == EOS_IDX:
            break
        if next_token == PAD_IDX:
            if finished:
                break
            continue

        char = IDX_TO_CHAR.get(next_token, "?")
        yield char

        tgt = torch.cat([tgt, torch.tensor([[next_token]], dtype=torch.long, device=device)], dim=1)


def chat():
    """Interactive chat with CurioS2S.

    Loads saved weights and lets you ask questions.
    Uses streaming output — characters appear one by one.
    Type 'quit' or 'exit' to leave.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = load_model(device=device)
    print("CurioS2S loaded! Ask me anything. (streaming output)\n")

    samples = load_samples()
    known_questions = {q for q, _ in samples}

    while True:
        try:
            question = input("You> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if question in ("quit", "exit", "q"):
            print("Bye!")
            break

        if not question:
            continue

        # --- Streaming output ---
        sys.stdout.write("CurioS2S> ")
        sys.stdout.flush()
        answer = ""
        for char in ask_stream(model, question, device=device):
            sys.stdout.write(char)
            sys.stdout.flush()
            answer += char
        sys.stdout.write("\n")

        if question in known_questions:
            expected = ""
            for q, a in samples:
                if q == question:
                    expected = a
                    break
            marker = "OK" if answer == expected else "MISS"
            print(f"  [{marker}, expected: {expected}]")
        print()


if __name__ == "__main__":
    chat()
