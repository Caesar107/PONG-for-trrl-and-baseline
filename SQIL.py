"""This is the runner of using TRRL to infer the reward functions and the optimal policy

"""
import pandas as pd
import tqdm
import datetime
import arguments
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import policies, MlpPolicy
from stable_baselines3.common import (
    base_class,
    distributions,
    on_policy_algorithm,
    policies,
    vec_env,
    evaluation
)
import copy
from imitation.util.util import make_vec_env
from imitation.algorithms.adversarial.common import AdversarialTrainer
from imitation.algorithms.adversarial.gail import GAIL
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

import gymnasium as gym
from reward_function import RwdFromRwdNet, RewardNet
from reward_function import BasicRewardNet
import rollouts
import os
import torch.utils.tensorboard as tb
from imitation.algorithms.sqil import SQIL
# from sqil import SQIL
from trrl import TRRL
from BC import BC
from typing import (
    List,
)
import torch.nn.functional as F
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from imitation.util import util
import logging
from imitation.policies.serialize import load_policy

logging.basicConfig(level=logging.WARNING)
class NullLogger(logging.Logger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def handle(self, record):
        # 忽略所有日志记录
        pass

# 创建空日志记录器实例
null_logger = NullLogger(name="null")



arglist = arguments.parse_args()

rng = np.random.default_rng(arglist.seed)
env = VecTransposeImage(make_vec_env(
    arglist.env_name,
    n_envs=arglist.n_env,
    rng=rng,
    #post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # for computing rollouts
))

print(arglist.env_name)

# # 定义专家模型路径
# expert_model_path = "/kaggle/working/ppo_for_breakout.zip"

# def train_expert():
#     """加载保存的专家策略"""
#     if not os.path.exists(expert_model_path):
#         raise FileNotFoundError(f"Expert model not found at {expert_model_path}")
#     print("Loading expert model.")
#     expert = PPO.load(
#         expert_model_path,
#         env=env,
#         #weights_only=True,
#         custom_objects={
#             'observation_space': env.observation_space,
#             'action_space': env.action_space
#         }
#     )
#     return expert


def train_expert():
    """Train an expert policy using PPO."""
    print("Training an expert.")
    expert = PPO(
        policy="MlpPolicy",
        env=env,
        seed=0,
        batch_size=min,
        ent_coef=arglist.ent_coef,
        learning_rate=arglist.lr,
        gamma=arglist.discount,
        n_epochs=5,
        n_steps=min,
    )
    expert.learn(100_000)  # Train for a sufficient number of steps
    return expert

def sample_expert_transitions(expert: policies):
    print("Sampling expert transitions.")
    trajs = rollouts.generate_trajectories(
        expert,
        env,
        rollouts.make_sample_until(min_timesteps=10000, min_episodes=100),
        rng=rng,
        starting_state= None, #np.array([0.1, 0.1, 0.1, 0.1]),
        starting_action=None, #np.array([[1,], [1,],], dtype=np.integer)
    )

    return rollouts.flatten_trajectories(trajs)
    #return rollouts

expert = train_expert()  # uncomment to train your own expert

mean_reward, std_reward = evaluate_policy(model=expert, env=env)
print("Average reward of the expert is evaluated at: " + str(mean_reward) + ',' + str(std_reward) + '.')

transitions = sample_expert_transitions(expert)

print("Number of transitions in demonstrations: " + str(transitions.obs.shape[0]) + ".")

rwd_net = BasicRewardNet(env.unwrapped.envs[0].unwrapped.observation_space, env.unwrapped.envs[0].unwrapped.action_space)

if arglist.device == 'cpu':
    DEVICE = torch.device('cpu')
elif arglist.device == 'gpu' and torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
elif arglist.device == 'gpu' and not torch.cuda.is_available():
    DEVICE = torch.device('cpu')
    print("Cuda is not available, run on CPU instead.")
else:
    DEVICE = torch.device('cpu')
    print("The intended device is not supported, run on CPU instead.")
print(DEVICE)

# Initialize TensorBoard logger
writer = tb.SummaryWriter(log_dir="logs/SQIL", flush_secs=1)

def _evaluate_policy(sqil_trainer, env) -> float:
    """Evalute the true expected return of the learned policy under the original environment.

    :return: The true expected return of the learning policy.
    """

    mean_reward, std_reward = evaluation.evaluate_policy(model=sqil_trainer.policy, env=env)


    return mean_reward


def expert_kl(sqil_trainer, expert, transitions) -> float:
    """KL divergence between the expert and the current policy.
    A Stablebaseline3-format expert policy is required.

    :return: The average KL divergence between the the expert policy and the current policy
    """
    obs = copy.deepcopy(transitions.obs)
    acts = copy.deepcopy(transitions.acts)

    obs_th = torch.as_tensor(obs, device='cuda:0')
    acts_th = torch.as_tensor(acts, device='cuda:0')

    # 确保模型的权重在同一设备上
    sqil_trainer.policy.to('cuda:0')
    expert.policy.to('cuda:0')

    target_values, target_log_prob, target_entropy = expert.policy.evaluate_actions(obs_th, acts_th)

    with torch.no_grad():
        q_values_sqil = sqil_trainer.policy.q_net(obs_th)
        probs_sqil = F.softmax(q_values_sqil, dim=1)

        # 选择与动作 acts_th 对应的概率
    probs_sqil_selected = probs_sqil.gather(1, acts_th.long().unsqueeze(-1)).squeeze(-1)

    kl_div = torch.mean(torch.dot(torch.exp(target_log_prob), target_log_prob - probs_sqil_selected.log()))
    return (float(kl_div))

sqil_trainer = SQIL(
    venv=env,
    demonstrations=transitions,
    policy="MlpPolicy",
)
total_timesteps = 3000000  # 总训练步数
eval_interval = 300  # 每隔多少步测试一次

log = 0
# 训练和测试循环
for timestep in tqdm.tqdm(range(0, total_timesteps, eval_interval)):
    # 训练模型

    sqil_trainer.train(total_timesteps=eval_interval)
    log += 1

    kl = expert_kl(sqil_trainer, expert, transitions)

    evaluate = _evaluate_policy(sqil_trainer, env)
    writer.add_scalar("Valid/distance", kl, log)
    writer.add_scalar("Valid/reward", evaluate, log)







