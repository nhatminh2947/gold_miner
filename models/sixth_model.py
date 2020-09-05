import numpy as np
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils import try_import_torch
from ray.rllib.utils.annotations import override

torch, nn = try_import_torch()


class SixthModel(RecurrentNetwork, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, in_channels):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.shared_conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=1
            ),
            nn.SELU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=1
            ),
            nn.SELU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=1
            ),
            nn.SELU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=1
            ),
            nn.SELU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1
            ),
            nn.SELU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1
            ),
            nn.SELU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1
            ),
            nn.SELU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1
            ),
            nn.SELU(),
            nn.Flatten()  # 1 * 13 * 64 = 832
        )

        self.shared_fc_layers = nn.Sequential(
            nn.Linear(832 + 6 + 3, 512),
            nn.SELU(),
            nn.Linear(512, 256),
            nn.SELU(),
            nn.Linear(256, 128)
        )

        self.lstm = nn.LSTM(128, 128, batch_first=True)

        self.actor_layers = nn.Sequential(
            nn.Linear(128, 6)
        )

        self.critic_layers = nn.Sequential(
            nn.Linear(128, 1)
        )

        self._features = None

    @override(ModelV2)
    def get_initial_state(self):
        # Place hidden states on same device as model.
        h = [
            torch.zeros(128, dtype=torch.float),
            torch.zeros(128, dtype=torch.float)
        ]
        return h

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"]["conv_features"]
        x = self.shared_conv_layers(x)
        x = torch.cat((x, input_dict["obs"]["fc_features"]), dim=1)
        x = self.shared_fc_layers(x)

        output, new_state = self.forward_rnn(
            add_time_dimension(x.float(), seq_lens, framework="torch"),
            state,
            seq_lens
        )

        return torch.reshape(output, [-1, self.num_outputs]), new_state

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        self._features, [h, c] = self.lstm(
            inputs,
            [torch.unsqueeze(state[0], 0),
             torch.unsqueeze(state[1], 0)]
        )

        action_out = self.actor_layers(self._features)
        return action_out, [torch.squeeze(h, 0), torch.squeeze(c, 0)]

    def value_function(self):
        return torch.reshape(self.critic_layers(self._features), [-1])
