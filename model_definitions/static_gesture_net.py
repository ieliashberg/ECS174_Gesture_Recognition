import torch.nn as nn

class StaticGestureNet(nn.Module):
        def __init__(self, input_dimension, num_classes):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dimension, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes),
            )

        def forward(self, x):
            return self.model(x)