from ray.rllib.agents.a3c import A2CTrainer
from ray.rllib.agents.ddpg import TD3Trainer
from ray.rllib.agents.impala import ImpalaTrainer
from ray.rllib.agents.marwil import MARWILTrainer
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.sac import SACTrainer

PPO = 'ppo'
A2C = 'a2c'
DDPG = 'ddpg'
TD3 = 'td3'
SAC = 'sac'
IMPALA = "impala"
MARWIL = 'marwil'

ALGOS = {
    PPO: PPOTrainer,
    A2C: A2CTrainer,
    TD3: TD3Trainer,
    SAC: SACTrainer,
    IMPALA: ImpalaTrainer,
    MARWIL: MARWILTrainer,
}

REWARD_CONT = 'reward_cont_mean'
REWARD_CONT_EVAL = 'reward_cont_eval_mean'
REWARD_HOST = 'reward_host_mean'
REWARD_HOST_EVAL = 'reward_host_eval_mean'
RCAR_DIST = 'rcar_dist_mean'
RCAR_DIST_EVAL = 'rcar_dist_eval_mean'

OLD_REWARD_CONT = 'policy_reward_mean_cont'
OLD_REWARD_HOST = 'policy_reward_mean_host'

POSITIVE_METRICS = {
    RCAR_DIST: 'RCAR distance',
    RCAR_DIST_EVAL: 'RCAR distance',
}

NEGATIVE_METRICS = {
    REWARD_CONT:        r'$\mathrm{L}_Q(x, Q(\cdot \mid y))$',
    REWARD_CONT_EVAL:   r'$\mathrm{L}_Q(x, Q(\cdot \mid y))$',
    REWARD_HOST:        r'$\mathrm{L}_P(x, Q(\cdot \mid y))$',
    REWARD_HOST_EVAL:   r'$\mathrm{L}_P(x, Q(\cdot \mid y))$',
    
    OLD_REWARD_CONT:    r'$\mathrm{L}_Q(x, Q(\cdot \mid y))$',
    OLD_REWARD_HOST:    r'$\mathrm{L}_P(x, Q(\cdot \mid y))$',
}

OLD_METRICS = {
    REWARD_CONT: OLD_REWARD_CONT,
    REWARD_HOST: OLD_REWARD_HOST,
}

ALL_METRICS = {**POSITIVE_METRICS, **NEGATIVE_METRICS}
