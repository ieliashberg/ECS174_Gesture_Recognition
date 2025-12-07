import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class DynamicGestureNet(nn.Module):
    def __init__(self, input_dim, num_classes, hidden = 128, layers = 2, bidir=True, dropout = 0.2):
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
      self.head = nn.Sequential(
         nn.LayerNorm(feat_dim),
         nn.Linear(feat_dim, num_classes),
      )

    def forward(self, x, lengths= None):
        if lengths is not None:
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, h_n = self.gru(packed)
        else:
            _, h_n = self.gru(x)

        if self.bidir:
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h_last = h_n[-1]
        return self.head(h_last)
