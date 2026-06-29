import os
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
)

CHECKPOINT_PATH = os.path.join("checkpoints", "curios2s.pt")


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


def chat():
    """Interactive chat with CurioS2S.

    Loads saved weights and lets you ask questions.
    Type 'quit' or 'exit' to leave.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = load_model(device=device)
    print("CurioS2S loaded! Ask me anything.\n")

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

        answer = ask(model, question, device=device)

        if question in known_questions:
            expected = ""
            for q, a in samples:
                if q == question:
                    expected = a
                    break
            marker = "OK" if answer == expected else "MISS"
            print(f"CurioS2S> {answer}  [{marker}, expected: {expected}]")
        else:
            print(f"CurioS2S> {answer}")
        print()


if __name__ == "__main__":
    chat()
