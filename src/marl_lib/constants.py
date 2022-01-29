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
