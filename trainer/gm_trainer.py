from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy, KLCoeffMixin, ValueNetworkMixin
from ray.rllib.policy.torch_policy import LearningRateSchedule, EntropyCoeffSchedule
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.schedules import ConstantSchedule

tf = try_import_tf()


class GoldMinerMixin:
    def __init__(self, config):
        self.config = config

    def update_lr_schedule(self, lr):
        self.lr_schedule = ConstantSchedule(lr, framework=None)

    def update_clip_param(self, clip_param):
        self.config["clip_param"] = clip_param

    def update_entropy(self, entropy_coeff):
        self.entropy_coeff_schedule = ConstantSchedule(entropy_coeff, framework="torch")


def setup_mixins(policy, obs_space, action_space, config):
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                  config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])
    GoldMinerMixin.__init__(policy, config)


GoldMinerPolicy = PPOTorchPolicy.with_updates(
    name="Custom_Policy",
    before_init=setup_mixins,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin,
        GoldMinerMixin
    ]
)


def get_policy_class(config):
    return GoldMinerPolicy


GoldMinerTrainer = PPOTrainer.with_updates(
    name="GoldMinerTrainer",
    default_policy=GoldMinerPolicy,
    get_policy_class=get_policy_class,
)
