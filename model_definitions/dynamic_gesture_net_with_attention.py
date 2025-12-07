import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# --------------------------------------------------------------------------- #
# TEMPORAL ATTENTION MODULE
# --------------------------------------------------------------------------- #
class TemporalAttention(nn.Module):
    """
    Computes attention weights over time for each frame in GRU output
    feats: (B, T, F) -> weighted sum vector (B, F)
    """
    def __init__(self, feat_dim):
        super().__init__()
        self.score = nn.Linear(feat_dim, 1)

    def forward(self, feats, lengths=None):
        # feats: (B, T, F)
        B, T, F = feats.shape

        # compute unnormalized attention weights
        w = self.score(feats).squeeze(-1)  # (B, T)

        # mask out padded timesteps
        if lengths is not None:
            mask = torch.arange(T)[None, :].to(feats.device) < lengths[:, None]
            w = w.masked_fill(~mask, float('-inf'))

        # normalized attention weights
        a = torch.softmax(w, dim=1)  # (B, T)

        # weighted feature aggregation
        return torch.sum(feats * a.unsqueeze(-1), dim=1)  # (B, F)


# --------------------------------------------------------------------------- #
# DYNAMIC GESTURE MODEL WITH TEMPORAL ATTENTION
# --------------------------------------------------------------------------- #
class DynamicGestureNet(nn.Module):
    def __init__(self, input_dim, num_classes,
                 hidden=128, layers=2, bidir=True, dropout=0.2):
        super().__init__()
        self.hidden = hidden
        self.layers = layers
        self.bidir = bidir

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidir,
            dropout=dropout if layers > 1 else 0.0,
        )

        feat_dim = hidden * (2 if bidir else 1)
        self.attn = TemporalAttention(feat_dim)

        self.norm = nn.LayerNorm(feat_dim)
        self.dropout = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat_dim // 2, num_classes)
        )

    def forward(self, x, lengths=None):
        # pack sequences for GRU efficiency
        if lengths is not None:
            packed = pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.gru(packed)
            feats, _ = pad_packed_sequence(packed_out, batch_first=True)
        else:
            feats, _ = self.gru(x)

        # feats shape: (B, T, F)
        h = self.attn(feats, lengths)  # temporal attention
        h = self.dropout(self.norm(h))
        return self.head(h)
