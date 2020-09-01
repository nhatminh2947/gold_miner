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

        self.shared_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1
            ),
            nn.ReLU(),
            nn.Flatten(),  # 1 * 13 * 64 = 832
            nn.Linear(832, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # last layer + one hot last action + last reward + energies + position (x, y)
        self.lstm = nn.LSTM(128 + 6 + 3, 128, batch_first=True)

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
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        x = input_dict["obs"]["conv_features"]
        x = self.shared_layers(x)

        if type(input_dict["prev_actions"]) != torch.Tensor:
            prev_actions = np.array(input_dict["prev_actions"], dtype=np.int)
        else:
            prev_actions = np.array(input_dict["prev_actions"].cpu().numpy(), dtype=np.int)
        prev_actions = np.expand_dims(prev_actions, 0)

        one_hot_prev_actions = torch.cat(
            [nn.functional.one_hot(torch.tensor(a), 6) for a in prev_actions],
            axis=-1
        )

        x = torch.cat((x, input_dict["obs"]["fc_features"], one_hot_prev_actions.float().to(device)),
                      dim=1)

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
