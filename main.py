"""This is the runner of using TRRL to infer the reward functions and the optimal policy"""

import os
import arguments
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecTransposeImage
from imitation.util.util import make_vec_env
from reward_function import BasicRewardNet
import rollouts
from trrl import TRRL

arglist = arguments.parse_args()

# 设置环境名称
arglist.env_name = 'Pong-v4'#'Breakout-v4'  # 或 'Pong-v4'
rng = np.random.default_rng(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
min=16
max =64

# 环境初始化
rng = np.random.default_rng(0)
env = VecTransposeImage(make_vec_env(
    arglist.env_name,
    n_envs=arglist.n_env,
    rng=rng,
))

print(f"Environment set to: {arglist.env_name}")

# 定义专家模型路径
expert_model_path = "/kaggle/working/ppo_for_breakout.zip"

def load_expert_model():
    """加载保存的专家策略"""
    if not os.path.exists(expert_model_path):
        raise FileNotFoundError(f"Expert model not found at {expert_model_path}")
    print("Loading expert model.")
    expert = PPO.load(
        expert_model_path,
        env=env,
        #weights_only=True,
        custom_objects={
            'observation_space': env.observation_space,
            'action_space': env.action_space
        }
    )
    return expert

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

def sample_expert_transitions(expert):
    """从专家策略中采样轨迹"""
    print("Sampling expert transitions.")
    trajs = rollouts.generate_trajectories(
        expert,
        env,
        rollouts.make_sample_until(min_timesteps=None, min_episodes=60),
        rng=rng,
        starting_state= None, #np.array([0.1, 0.1, 0.1, 0.1]),
        starting_action=None, #np.array([[1,], [1,],], dtype=np.integer)
    )
    return rollouts.flatten_trajectories(trajs)

# 加载专家策略
expert = train_expert()

# 评估专家策略
mean_reward, std_reward = evaluate_policy(model=expert, env=env, n_eval_episodes=10)
print(f"Average reward of the expert: {mean_reward}, Std reward: {std_reward}.")

# 从专家策略采样轨迹
transitions = sample_expert_transitions(expert)
print(f"Number of transitions in demonstrations: {transitions.obs.shape[0]}.")

# 定义奖励网络
rwd_net = BasicRewardNet(env.unwrapped.envs[0].unwrapped.observation_space, env.unwrapped.envs[0].unwrapped.action_space)

# 选择设备
DEVICE = torch.device('cuda:0' if arglist.device == 'gpu' and torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cpu' and arglist.device == 'gpu':
    print("Cuda is not available, running on CPU instead.")

print(f"Environment observation space before TRRL: {env.observation_space}")

# 初始化 TRRL
trrl_trainer = TRRL(
    venv=env,
    expert_policy=expert,
    demonstrations=transitions,
    demo_batch_size=arglist.demo_batch_size,
    reward_net=rwd_net,
    discount=arglist.discount,
    avg_diff_coef=arglist.avg_reward_diff_coef,
    l2_norm_coef=arglist.avg_reward_diff_coef,
    l2_norm_upper_bound=arglist.l2_norm_upper_bound,
    ent_coef=arglist.ent_coef,
    device=DEVICE,
    n_policy_updates_per_round=100,
    n_reward_updates_per_round=2,
    n_episodes_adv_fn_est=1,
    n_timesteps_adv_fn_est=5
)

print("Starting reward learning.")
trrl_trainer.train(n_rounds=arglist.n_runs)
