import os
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .model import CurioNet
from .data import (
    WikiText2Dataset,
    get_dataloader,
    VOCAB_SIZE,
    PAD_IDX,
    BOS_IDX,
    EOS_IDX,
)


def _save_checkpoint(model, config, history, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "history": history,
    }, path)


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class WarmupScheduler:
    def __init__(self, optimizer, lr: float, warmup_epochs: int = 5):
        self.optimizer = optimizer
        self.lr = lr
        self.warmup_epochs = warmup_epochs

    def step(self, epoch: int):
        if epoch <= self.warmup_epochs:
            scale = epoch / self.warmup_epochs
        else:
            scale = 1.0
        current_lr = self.lr * scale
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr
        return current_lr


def train_curionet(
    epochs: int = 30,
    batch_size: int = 64,
    seq_len: int = 64,
    dim: int = 46,
    num_layers: int = 2,
    lr: float = 3e-4,
    device: str = "cuda",
    plot_dir: str = "plots",
    checkpoint_dir: str = "checkpoints",
    patience: int = 4,
    warmup_epochs: int = 5,
) -> dict:
    """Train CurioNet on WikiText-2 and save checkpoint for chat.

    Configured for ~300K parameters (dim=46, num_layers=2).
    Runs on GPU (CUDA required).
    """
    if not torch.cuda.is_available() and device == "cuda":
        raise RuntimeError("CUDA is required. No GPU detected.")

    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Loading WikiText-2...")

    train_ds = WikiText2Dataset(split="train", seq_len=seq_len, max_chars=500000)
    val_ds = WikiText2Dataset(split="validation", seq_len=seq_len, max_chars=100000)
    train_loader = get_dataloader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")

    model_config = {
        "src_vocab_size": VOCAB_SIZE,
        "tgt_vocab_size": VOCAB_SIZE,
        "dim": dim,
        "num_layers": num_layers,
        "dropout": 0.1,
        "padding_idx": PAD_IDX,
    }

    model = CurioNet(
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
        dim=dim,
        num_layers=num_layers,
        dropout=0.1,
        padding_idx=PAD_IDX,
        max_len=seq_len + 4,
    ).to(device)

    params = _count_params(model)
    print(f"Parameters: {params:,}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = WarmupScheduler(optimizer, lr=lr, warmup_epochs=warmup_epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    history = {"epoch": [], "train_loss": [], "train_ppl": [], "val_loss": [], "val_ppl": []}
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        current_lr = scheduler.step(epoch)
        model.train()
        total_loss = 0.0
        total_tokens = 0
        epoch_start = time.time()

        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            dec_input = tgt[:, :-1]
            dec_target = tgt[:, 1:]

            logits = model(src, dec_input)
            loss = criterion(logits.reshape(-1, logits.size(-1)), dec_target.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            mask = dec_target.ne(PAD_IDX)
            total_loss += loss.item() * mask.sum().item()
            total_tokens += mask.sum().item()

        train_loss = total_loss / max(total_tokens, 1)
        train_ppl = math.exp(min(train_loss, 20))

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_token_sum = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                dec_input = tgt[:, :-1]
                dec_target = tgt[:, 1:]
                logits = model(src, dec_input)
                loss = criterion(logits.reshape(-1, logits.size(-1)), dec_target.reshape(-1))
                mask = dec_target.ne(PAD_IDX)
                val_loss_sum += loss.item() * mask.sum().item()
                val_token_sum += mask.sum().item()

        val_loss = val_loss_sum / max(val_token_sum, 1)
        val_ppl = math.exp(min(val_loss, 20))
        epoch_time = time.time() - epoch_start

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_ppl"].append(train_ppl)
        history["val_loss"].append(val_loss)
        history["val_ppl"].append(val_ppl)

        print(f"Epoch {epoch:3d}/{epochs} ({epoch_time:.1f}s) | LR: {current_lr:.6f} "
              f"| Train: {train_loss:.4f} PPL: {train_ppl:.2f} "
              f"| Val: {val_loss:.4f} PPL: {val_ppl:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_path = os.path.join(checkpoint_dir, "curionet_best.pt")
            _save_checkpoint(model, model_config, history, best_path)
            print(f"  -> New best (val PPL: {val_ppl:.2f})")
        else:
            patience_counter += 1
            print(f"  -> No improvement ({patience_counter}/{patience})")

        latest_path = os.path.join(checkpoint_dir, "curionet_latest.pt")
        _save_checkpoint(model, model_config, history, latest_path)

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle("CurioNet — WikiText-2 Training", fontsize=14, fontweight="bold")

    axes[0].plot(history["epoch"], history["train_loss"], "b-o", ms=2, label="Train")
    axes[0].plot(history["epoch"], history["val_loss"], "r-s", ms=2, label="Val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["epoch"], history["train_ppl"], "b-o", ms=2, label="Train")
    axes[1].plot(history["epoch"], history["val_ppl"], "r-s", ms=2, label="Val")
    axes[1].set_title("Perplexity")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].bar(["Params"], [params], color="#2471a3")
    axes[2].set_title("Parameters")
    axes[2].text(0, params + 1000, f"{params:,}", ha="center")

    plt.tight_layout()
    plot_path = os.path.join(plot_dir, "curionet_wikitext2.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nPlot saved to: {plot_path}")
    print(f"Best model: {os.path.join(checkpoint_dir, 'curionet_best.pt')}")

    return {"model": model, "history": history, "params": params}


if __name__ == "__main__":
    train_curionet()
