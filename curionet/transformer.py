import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TransformerEncoderBlock(nn.Module):
    """Standard Transformer encoder block: multi-head attention + FFN."""

    def __init__(self, dim: int, num_heads: int = 4, ff_dim: int = None, dropout: float = 0.1):
        super().__init__()
        if ff_dim is None:
            ff_dim = dim * 4
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class TransformerDecoderBlock(nn.Module):
    """Standard Transformer decoder block: masked self-attn + cross-attn + FFN."""

    def __init__(self, dim: int, num_heads: int = 4, ff_dim: int = None, dropout: float = 0.1):
        super().__init__()
        if ff_dim is None:
            ff_dim = dim * 4
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor,
                tgt_mask: torch.Tensor = None, enc_mask: torch.Tensor = None,
                causal_mask: torch.Tensor = None) -> torch.Tensor:
        self_attn_out, _ = self.self_attn(x, x, x, attn_mask=causal_mask, key_padding_mask=tgt_mask)
        x = self.norm1(x + self_attn_out)

        cross_out, _ = self.cross_attn(x, enc_out, enc_out, key_padding_mask=enc_mask)
        x = self.norm2(x + cross_out)

        x = self.norm3(x + self.ffn(x))
        return x


class TransformerSeq2Seq(nn.Module):
    """Standard Transformer Seq2Seq for comparison with CurioNet.

    Same parameter budget as CurioNet for fair comparison.
    Uses multi-head attention in every layer (the standard approach).

    Args:
        src_vocab_size: Source vocabulary size.
        tgt_vocab_size: Target vocabulary size.
        dim: Hidden dimension.
        num_layers: Number of encoder/decoder layers.
        num_heads: Number of attention heads.
        ff_dim: Feed-forward dimension.
        dropout: Dropout rate.
        max_len: Max sequence length.
        padding_idx: Padding token index.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 4,
        ff_dim: int = None,
        dropout: float = 0.1,
        max_len: int = 256,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.dim = dim

        # Embeddings
        self.token_emb = nn.Embedding(src_vocab_size, dim, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding(max_len, dim, padding_idx=padding_idx)
        self.scale = math.sqrt(dim)
        self.emb_norm = nn.LayerNorm(dim)
        self.emb_dropout = nn.Dropout(dropout)

        # Decoder embeddings
        self.tgt_token_emb = nn.Embedding(tgt_vocab_size, dim, padding_idx=padding_idx)
        self.tgt_pos_emb = nn.Embedding(max_len, dim, padding_idx=padding_idx)
        self.tgt_emb_norm = nn.LayerNorm(dim)
        self.tgt_emb_dropout = nn.Dropout(dropout)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(dim, num_heads, ff_dim, dropout) for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderBlock(dim, num_heads, ff_dim, dropout) for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(dim, tgt_vocab_size)

    def encode(self, src: torch.Tensor) -> tuple:
        batch, seq_len = src.shape
        mask = src.ne(self.padding_idx)
        pos = torch.arange(seq_len, device=src.device).unsqueeze(0).expand(batch, seq_len)
        x = self.token_emb(src) * self.scale + self.pos_emb(pos)
        x = self.emb_norm(x)
        x = self.emb_dropout(x)

        enc_mask = ~mask
        for layer in self.encoder_layers:
            x = layer(x, enc_mask)

        return x, mask

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        enc_out, enc_mask = self.encode(src)

        batch, tgt_len = tgt.shape
        tgt_mask = tgt.ne(self.padding_idx)
        pos = torch.arange(tgt_len, device=tgt.device).unsqueeze(0).expand(batch, tgt_len)
        x = self.tgt_token_emb(tgt) * self.scale + self.tgt_pos_emb(pos)
        x = self.tgt_emb_norm(x)
        x = self.tgt_emb_dropout(x)

        causal = torch.triu(torch.ones(tgt_len, tgt_len, device=tgt.device), diagonal=1).bool()
        causal_mask = causal.unsqueeze(0).expand(batch * 4, -1, -1).reshape(batch, 4, tgt_len, tgt_len)
        causal_mask = causal_mask[:, 0]

        enc_key_mask = ~enc_mask
        tgt_key_mask = ~tgt_mask

        for layer in self.decoder_layers:
            x = layer(x, enc_out, tgt_key_mask, enc_key_mask, causal_mask)

        return self.output_proj(x)

    @torch.no_grad()
    def generate(self, src: torch.Tensor, max_len: int = 50, bos_idx: int = 1, eos_idx: int = 2) -> torch.Tensor:
        """Greedy decoding for inference."""
        self.eval()
        batch = src.size(0)
        device = src.device
        enc_out, enc_mask = self.encode(src)

        tgt = torch.full((batch, 1), bos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            batch_t, tgt_len = tgt.shape
            pos = torch.arange(tgt_len, device=device).unsqueeze(0).expand(batch_t, tgt_len)
            x = self.tgt_token_emb(tgt) * self.scale + self.tgt_pos_emb(pos)
            x = self.tgt_emb_norm(x)

            causal = torch.triu(torch.ones(tgt_len, tgt_len, device=device), diagonal=1).bool()
            causal_mask = causal.unsqueeze(0).expand(batch_t * 4, -1, -1).reshape(batch_t, 4, tgt_len, tgt_len)[:, 0]

            enc_key_mask = ~enc_mask
            for layer in self.decoder_layers:
                x = layer(x, enc_out, None, enc_key_mask, causal_mask)

            logits = self.output_proj(x)
            next_token = logits[:, -1, :].argmax(dim=-1)
            next_token = next_token.masked_fill(finished, self.padding_idx)
            tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)
            finished = finished | (next_token == eos_idx)
            if finished.all():
                break

        return tgt
