import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .curiosity import CuriosityBlock


class TinyAttention(nn.Module):
    """Minimal single-head attention — used sparingly in CurioNet.

    This is NOT the primary mechanism. It's a small residual that
    lets the model share basic cross-sequence information every N
    layers. Curiosity does the heavy lifting; this just helps.

    Single-head, no multi-head complexity, minimal parameters.
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        self.scale = 1.0 / math.sqrt(dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        ctx = torch.matmul(attn, v)
        ctx = self.out(ctx)

        return self.norm(x + ctx)


class CurioNetEncoder(nn.Module):
    """CurioNet Encoder — curiosity layers with sparse tiny attention.

    Architecture:
        [CuriosityBlock] × N, with TinyAttention every `attn_every` layers.

    The majority of processing is curiosity (wonder→process→insight).
    Tiny attention appears every few layers for basic info sharing.

    Args:
        vocab_size: Vocabulary size.
        dim: Hidden dimension.
        num_layers: Number of curiosity blocks.
        ff_dim: Feed-forward dimension.
        num_wonder_convs: Conv layers per curiosity block.
        kernel_size: Conv kernel size.
        dropout: Dropout rate.
        curiosity_budget: Wonder cycles per layer.
        attn_every: Insert tiny attention every N curiosity blocks.
        max_len: Max sequence length.
        padding_idx: Padding token index.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 128,
        num_layers: int = 6,
        ff_dim: int = None,
        num_wonder_convs: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1,
        curiosity_budget: int = 1,
        attn_every: int = 3,
        max_len: int = 256,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.dim = dim
        self.padding_idx = padding_idx
        self.token_emb = nn.Embedding(vocab_size, dim, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding(max_len, dim, padding_idx=padding_idx)
        self.scale = math.sqrt(dim)
        self.emb_norm = nn.LayerNorm(dim)
        self.emb_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList()
        self.tiny_attns = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(CuriosityBlock(
                dim, ff_dim, num_wonder_convs, kernel_size, dropout, curiosity_budget
            ))
            if (i + 1) % attn_every == 0 and (i + 1) < num_layers:
                self.tiny_attns.append(TinyAttention(dim, dropout))
            else:
                self.tiny_attns.append(None)

    def forward(self, src: torch.Tensor) -> tuple:
        # src: (batch, src_len)
        batch, seq_len = src.shape
        mask = src.ne(self.padding_idx)

        pos = torch.arange(seq_len, device=src.device).unsqueeze(0).expand(batch, seq_len)
        x = self.token_emb(src) * self.scale + self.pos_emb(pos)
        x = self.emb_norm(x)
        x = self.emb_dropout(x)

        wonders = []
        for i, block in enumerate(self.blocks):
            x, wonder = block(x)
            wonders.append(wonder)
            if self.tiny_attns[i] is not None:
                x = self.tiny_attns[i](x, mask)

        return x, mask, wonders


class CurioNetDecoder(nn.Module):
    """CurioNet Decoder — curiosity layers with cross-attention + sparse tiny attention.

    Architecture:
        [CuriosityBlock + CrossAttention] × N, with TinyAttention every `attn_every` layers.

    Cross-attention is the only "full" attention — needed for encoder-decoder
    info flow. Even this is single-head and minimal.

    Args:
        vocab_size: Vocabulary size.
        dim: Hidden dimension.
        num_layers: Number of curiosity blocks.
        ff_dim: Feed-forward dimension.
        num_wonder_convs: Conv layers per curiosity block.
        kernel_size: Conv kernel size.
        dropout: Dropout rate.
        curiosity_budget: Wonder cycles per layer.
        attn_every: Insert tiny self-attention every N curiosity blocks.
        max_len: Max sequence length.
        padding_idx: Padding token index.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 128,
        num_layers: int = 6,
        ff_dim: int = None,
        num_wonder_convs: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1,
        curiosity_budget: int = 1,
        attn_every: int = 3,
        max_len: int = 256,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.dim = dim
        self.padding_idx = padding_idx
        self.token_emb = nn.Embedding(vocab_size, dim, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding(max_len, dim, padding_idx=padding_idx)
        self.scale = math.sqrt(dim)
        self.emb_norm = nn.LayerNorm(dim)
        self.emb_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList()
        self.tiny_attns = nn.ModuleList()
        self.cross_attns = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(CuriosityBlock(
                dim, ff_dim, num_wonder_convs, kernel_size, dropout, curiosity_budget
            ))
            self.cross_attns.append(nn.MultiheadAttention(dim, num_heads=1, dropout=dropout, batch_first=True))
            if (i + 1) % attn_every == 0 and (i + 1) < num_layers:
                self.tiny_attns.append(TinyAttention(dim, dropout))
            else:
                self.tiny_attns.append(None)

        self.output_proj = nn.Linear(dim, vocab_size)

    def forward(self, tgt: torch.Tensor, enc_out: tuple) -> torch.Tensor:
        # tgt: (batch, tgt_len)
        # enc_out: (enc_hidden, enc_mask, enc_wonders)
        enc_hidden, enc_mask, _ = enc_out

        batch, seq_len = tgt.shape
        tgt_mask = tgt.ne(self.padding_idx)

        pos = torch.arange(seq_len, device=tgt.device).unsqueeze(0).expand(batch, seq_len)
        x = self.token_emb(tgt) * self.scale + self.pos_emb(pos)
        x = self.emb_norm(x)
        x = self.emb_dropout(x)

        # Causal mask for self-attention (if used)
        causal = torch.triu(torch.ones(seq_len, seq_len, device=tgt.device), diagonal=1).bool()

        for i, block in enumerate(self.blocks):
            x, wonder = block(x)

            # Cross-attention to encoder (minimal, single-head)
            ca_out, _ = self.cross_attns[i](
                x, enc_hidden, enc_hidden,
                key_padding_mask=~enc_mask,
            )
            x = x + ca_out

            # Tiny self-attention (sparse)
            if self.tiny_attns[i] is not None:
                x = self.tiny_attns[i](x, tgt_mask & ~causal.unsqueeze(0).expand(batch, -1, -1)[:, 0])

        logits = self.output_proj(x)
        return logits


class CurioNet(nn.Module):
    """CurioNet: Curiosity-driven Sequence-to-Sequence model.

    Unlike Transformers where attention is the primary mechanism,
    CurioNet uses CURIOSITY as its core. Curiosity layers generate
    wonder states, process them through 'thinking' conv layers, and
    extract insights — this is the main forward path.

    A tiny amount of attention is used sparingly:
    - TinyAttention every `attn_every` layers in encoder/decoder (single-head)
    - Cross-attention from decoder to encoder (single-head, minimal)

    This is NOT an attention-based model. It's a curiosity-based model
    with a small attention residual for basic info sharing.

    Comparison with Transformer:
    ┌─────────────────┬──────────────────┬──────────────────┐
    │                 │ Transformer      │ CurioNet         │
    ├─────────────────┼──────────────────┼──────────────────┤
    │ Primary mech    │ Multi-head attn  │ Curiosity layers │
    │ Core operation  │ Q·Kᵀ → weighted V│ Wonder→Think→Ins │
    │ Self-attention  │ Every layer      │ Every N layers   │
    │ Cross-attention │ Multi-head, every│ Single-head, min │
    │ Parameters      │ Heavy (Q,K,V,O)  │ Light (wonder)   │
    │ Info sharing    │ Re-weights exist │ Generates new    │
    └─────────────────┴──────────────────┴──────────────────┘

    Args:
        src_vocab_size: Source vocabulary size.
        tgt_vocab_size: Target vocabulary size.
        dim: Hidden dimension (default 128).
        num_layers: Curiosity blocks per encoder/decoder (default 6).
        ff_dim: Feed-forward dimension.
        num_wonder_convs: Conv layers per curiosity block (default 2).
        kernel_size: Conv kernel size (default 3).
        dropout: Dropout rate (default 0.1).
        curiosity_budget: Wonder cycles per layer (default 1).
        attn_every: Tiny attention every N layers (default 3).
        max_len: Max sequence length (default 256).
        padding_idx: Padding token index (default 0).
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dim: int = 128,
        num_layers: int = 6,
        ff_dim: int = None,
        num_wonder_convs: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1,
        curiosity_budget: int = 1,
        attn_every: int = 3,
        max_len: int = 256,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.encoder = CurioNetEncoder(
            src_vocab_size, dim, num_layers, ff_dim,
            num_wonder_convs, kernel_size, dropout,
            curiosity_budget, attn_every, max_len, padding_idx,
        )
        self.decoder = CurioNetDecoder(
            tgt_vocab_size, dim, num_layers, ff_dim,
            num_wonder_convs, kernel_size, dropout,
            curiosity_budget, attn_every, max_len, padding_idx,
        )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        enc_out = self.encoder(src)
        logits = self.decoder(tgt, enc_out)
        return logits

    @torch.no_grad()
    def generate(self, src: torch.Tensor, max_len: int = 50, bos_idx: int = 1, eos_idx: int = 2) -> torch.Tensor:
        """Greedy decoding for inference."""
        self.eval()
        batch = src.size(0)
        device = src.device
        enc_out = self.encoder(src)

        tgt = torch.full((batch, 1), bos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            logits = self.decoder(tgt, enc_out)
            next_token = logits[:, -1, :].argmax(dim=-1)
            next_token = next_token.masked_fill(finished, self.decoder.padding_idx)
            tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)
            finished = finished | (next_token == eos_idx)
            if finished.all():
                break

        return tgt
