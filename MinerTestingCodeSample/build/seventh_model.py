from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import try_import_torch
from ray.rllib.utils.annotations import override

torch, nn = try_import_torch()


class SeventhModel(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.shared_conv_layers = nn.Sequential(
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
            nn.Flatten()  # 1 * 13 * 64 = 832
        )

        self.shared_fc_layers = nn.Sequential(
            nn.Linear(860, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
        )

        self.actor_layers = nn.Sequential(
            nn.Linear(128, 6)
        )

        self.critic_layers = nn.Sequential(
            nn.Linear(128, 1)
        )

        self._shared_layer_out = None
        self._features = None

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"]["conv_features"]
        x = self.shared_conv_layers(x)
        x = torch.cat((x, input_dict["obs"]["fc_features"]), dim=1)

        self._shared_layer_out = self.shared_fc_layers(x)
        logits = self.actor_layers(self._shared_layer_out)
        return logits, state

    def predict(self, input_dict):
        x = self.forward(input_dict, [], None)[0]
        print(x)
        x = torch.argmax(x).item()
        print(x)
        return x