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
EXP_ENTROPY = 'expected_entropy_mean'
EXP_ENTROPY_EVAL = 'expected_entropy_eval_mean'

OLD_REWARD_CONT = 'policy_reward_mean/cont'
OLD_REWARD_HOST = 'policy_reward_mean/host'

PRETTY_NAMES = {
    REWARD_CONT: 'Contestant expected loss',  # r'$\mathrm{\mathbb{E}}[\mathrm{L}_C(x, Q)]$',
    REWARD_HOST: 'Host expected loss',  # r'$\mathrm{\mathbb{E}}[\mathrm{L}_H(x, Q)]$',
    RCAR_DIST: 'RCAR distance',
    EXP_ENTROPY: 'Expected generalised entropy'  # r'$\mathrm{\mathbb{E}}[\mathrm{H}_L_C (P)]$',
}

POSITIVE_METRICS = {
    RCAR_DIST: PRETTY_NAMES[RCAR_DIST],
    RCAR_DIST_EVAL: PRETTY_NAMES[RCAR_DIST],
    EXP_ENTROPY: PRETTY_NAMES[EXP_ENTROPY],
    EXP_ENTROPY_EVAL: PRETTY_NAMES[EXP_ENTROPY],
}

NEGATIVE_METRICS = {
    REWARD_CONT: PRETTY_NAMES[REWARD_CONT],
    REWARD_CONT_EVAL: PRETTY_NAMES[REWARD_CONT],
    REWARD_HOST: PRETTY_NAMES[REWARD_HOST],
    REWARD_HOST_EVAL: PRETTY_NAMES[REWARD_HOST],
    
    OLD_REWARD_CONT: PRETTY_NAMES[REWARD_CONT],
    OLD_REWARD_HOST: PRETTY_NAMES[REWARD_HOST],
}

OLD_METRICS = {
    REWARD_CONT: OLD_REWARD_CONT,
    REWARD_HOST: OLD_REWARD_HOST,
}

ALL_METRICS = {**POSITIVE_METRICS, **NEGATIVE_METRICS}
