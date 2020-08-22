import torch
from torch import nn

class FifthModel(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.shared_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=1
            ),
            nn.ELU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=1
            ),
            nn.ELU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=1
            ),
            nn.ELU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=1
            ),
            nn.ELU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=1
            ),
            nn.ELU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=1
            ),
            nn.ELU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=1
            ),
            nn.ELU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=1
            ),
            nn.ELU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1
            ),
            nn.ELU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1
            ),
            nn.ELU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1
            ),
            nn.ELU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1
            ),
            nn.ELU(),
            nn.Flatten(),  # 1 * 13 * 64 = 832
            nn.Linear(832, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
        )

        self.actor_layers = nn.Sequential(
            nn.Linear(128 + 2, 6)
        )

        self.critic_layers = nn.Sequential(
            nn.Linear(128 + 2, 1)
        )

        self._shared_layer_out = None
        self._features = None

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"]["conv_features"]
        x = self.shared_layers(x)

        self._shared_layer_out = torch.cat((x, input_dict["obs"]["fc_features"]), dim=1)
        logits = self.actor_layers(self._shared_layer_out)

        return logits, state

    def predict(self, input_dict):
        x = self.forward(input_dict, [], None)[0]
        print(x)
        return torch.argmax(x).item()

    def value_function(self):
        return torch.reshape(self.critic_layers(self._shared_layer_out), [-1])
