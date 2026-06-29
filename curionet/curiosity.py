import torch
import torch.nn as nn
import math


class WonderGenerator(nn.Module):
    """Generates wonder states — the model's internal curiosity.

    This is NOT attention: it doesn't weight or select from existing inputs.
    It GENERATES entirely new internal states — like a human
    who reads something and then wonders 'but what about...?'
    """

    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        self.expand = nn.Linear(dim, dim * expansion)
        self.act = nn.GELU()
        self.contract = nn.Linear(dim * expansion, dim)
        self.gate = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.expand(x)
        h = self.act(h)
        h = self.contract(h)
        g = torch.sigmoid(self.gate(x))
        wonder = h * g
        return self.norm(wonder)


class WonderMixer(nn.Module):
    """Global curiosity-driven token mixing — replaces convolutions.

    Unlike attention (Q·K^T → softmax → weighted V), WonderMixer:
    1. Takes wonder states (generated curiosity, not raw representations)
    2. Projects wonder into seeker/guide representations
    3. Computes curiosity affinity via SIGMOID (independent gating)
    4. Mixes token information based on this curiosity affinity

    Key differences from attention:
    - Operates on GENERATED wonder states, not raw token embeddings
    - Uses sigmoid (each token independently decides what to take)
    - No softmax (no competitive budget allocation)
    - Wonder drives everything — no separate Q/K/V on raw x

    This gives global receptive field (like Transformer) but through
    curiosity, not attention. Every token can access every other token.
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dim = dim

        self.wonder_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, wonder: torch.Tensor,
                mask: torch.Tensor = None, causal: bool = False) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        sg = self.wonder_proj(wonder)
        seeker, guide = sg.chunk(2, dim=-1)

        seeker = seeker.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        guide = guide.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        affinity = torch.matmul(seeker, guide.transpose(-2, -1)) * self.scale
        affinity = torch.sigmoid(affinity)

        if causal:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            affinity = affinity.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), 0.0)

        if mask is not None:
            pad_mask = (~mask).unsqueeze(1).unsqueeze(1)
            affinity = affinity.masked_fill(pad_mask, 0.0)

        affinity = self.dropout(affinity)

        x_heads = x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        mixed = torch.matmul(affinity, x_heads)
        mixed = mixed.transpose(1, 2).reshape(batch, seq_len, self.dim)

        out = self.out_proj(mixed)
        return self.norm(x + out)


class InsightExtractor(nn.Module):
    """Extracts insights from mixed wonder states.

    Insights are the 'aha!' moments — the model synthesizes
    what it learned from its curiosity into actionable knowledge.
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.act(x)
        x = self.dropout(x)
        return self.norm(x)


class CuriosityLayer(nn.Module):
    """A single curiosity layer — the primary building block of CurioNet.

    Replaces what an attention layer does in a Transformer:
    - Transformer: Q·K^T → softmax → weighted V (re-weighting existing info)
    - CurioNet: generate wonder → mix wonder globally → extract insights → integrate

    The curiosity layer GENERATES new internal states, MIXES them globally
    across the full sequence (not locally like convolutions), and EXTRACTS
    insights — this is the fundamental difference from both attention and CNNs.

    Pipeline:
    1. Generate wonder from current representation (point-wise)
    2. Mix wonder globally across tokens (WonderMixer — curiosity affinity)
    3. Extract insights from mixed wonder (point-wise)
    4. Integrate insights back via gated residual

    Args:
        dim: Hidden dimension.
        num_heads: Number of curiosity aspects for WonderMixer.
        dropout: Dropout rate.
        curiosity_budget: Wonder cycles per layer (higher = more curious).
        causal: If True, mixer uses causal masking (for decoder).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        curiosity_budget: int = 1,
        causal: bool = False,
    ):
        super().__init__()
        self.curiosity_budget = curiosity_budget
        self.causal = causal
        self.wonder_gen = WonderGenerator(dim)
        self.mixer = WonderMixer(dim, num_heads, dropout)
        self.insight = InsightExtractor(dim, dropout)
        self.integrate_gate = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> tuple:
        residual = x
        current = x
        last_wonder = None

        for _ in range(self.curiosity_budget):
            # 1. Generate wonder (point-wise curiosity)
            wonder = self.wonder_gen(current)
            last_wonder = wonder

            # 2. Mix wonder globally across tokens (curiosity-driven, NOT attention)
            current = self.mixer(current, wonder, mask=mask, causal=self.causal)

            # 3. Extract insights from mixed representation
            insights = self.insight(current)

            # 4. Integrate insights via gated residual
            current = current + insights

        integrated = self.integrate_gate(torch.cat([current, residual], dim=-1))
        output = self.norm(integrated + residual)

        return output, last_wonder


class CuriosityBlock(nn.Module):
    """Curiosity layer + feed-forward — like a Transformer block but curiosity-based.

    Transformer block:  [Attention → FFN]
    CurioNet block:     [Curiosity → FFN]

    The FFN is the same (Linear → GELU → Linear), but the core
    mechanism is curiosity instead of attention.

    Args:
        dim: Hidden dimension.
        ff_dim: Feed-forward expansion dimension.
        num_heads: Number of curiosity aspects for WonderMixer.
        dropout: Dropout rate.
        curiosity_budget: Wonder cycles.
        causal: If True, mixer uses causal masking (for decoder).
    """

    def __init__(
        self,
        dim: int,
        ff_dim: int = None,
        num_heads: int = 4,
        dropout: float = 0.1,
        curiosity_budget: int = 1,
        causal: bool = False,
    ):
        super().__init__()
        if ff_dim is None:
            ff_dim = dim * 4
        self.curiosity = CuriosityLayer(
            dim, num_heads, dropout, curiosity_budget, causal=causal
        )
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> tuple:
        x, wonder = self.curiosity(x, mask=mask)
        x = self.ffn_norm(x + self.ffn(x))
        return x, wonder
