"""
This is the runner of using GAIL as the baseline to infer the reward functions and the optimal policy
"""
import arguments
import os
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.algorithms.adversarial import gail
from imitation.util.util import make_vec_env
import torch.utils.tensorboard as tb
import rollouts
from reward_function import BasicRewardNet
import logging
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

# Remove default terminal logging
logger = logging.getLogger()
logger.handlers = []

# Parse arguments
arglist = arguments.parse_args()
rng = np.random.default_rng(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
min_batch_size = 16
max_batch_size = 64

def compute_kl_divergence(expert_policy, current_policy, observations, actions, device):
    """
    Computes the KL divergence between the expert policy and the current policy.

    Args:
        expert_policy: The expert policy (Stable-Baselines3 model object).
        current_policy: The current learned policy (Stable-Baselines3 model object).
        observations: Observations (numpy array or tensor).
        actions: Actions taken (numpy array or tensor).
        device: PyTorch device (e.g., 'cpu' or 'cuda').

    Returns:
        kl_divergence: The mean KL divergence.
    """
    # Convert observations and actions to tensors
    obs_th = torch.as_tensor(observations, device=device)
    acts_th = torch.as_tensor(actions, device=device)

    # Ensure both policies are on the same device
    expert_policy.policy.to(device)
    current_policy.to(device)

    # Get log probabilities from both policies
    input_values, input_log_prob, input_entropy = current_policy.evaluate_actions(obs_th, acts_th)
    target_values, target_log_prob, target_entropy = expert_policy.policy.evaluate_actions(obs_th, acts_th)

    # Compute KL divergence using TRRO's logic
    kl_divergence = torch.mean(torch.dot(torch.exp(target_log_prob), target_log_prob - input_log_prob)).item()

    return kl_divergence

# Create environment
# Create environment
env = VecTransposeImage(make_vec_env(
    arglist.env_name,
    n_envs=1,#arglist.n_env,
    rng=rng,
)
)
print(f"Environment set to: {arglist.env_name}")

# Initialize TensorBoard logger
writer = tb.SummaryWriter(log_dir="logs/GAIL", flush_secs=1)

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
        batch_size=min_batch_size,
        ent_coef=arglist.ent_coef,
        learning_rate=arglist.lr,
        gamma=arglist.discount,
        n_epochs=5,
        n_steps=min_batch_size,
    )
    expert.learn(100_000)  # Train for a sufficient number of steps
    return expert


def sample_expert_transitions(expert):
    """Sample transitions from the trained expert."""
    print("Sampling expert transitions.")
    trajs = rollouts.generate_trajectories(
        expert,
        env,
        rollouts.make_sample_until(min_timesteps=None, min_episodes=60),
        rng=rng,
        starting_state=None,
        starting_action=None,
    )
    return rollouts.flatten_trajectories(trajs)


# Train expert
expert = train_expert()

# Evaluate expert policy
mean_reward, std_reward = evaluate_policy(model=expert, env=env)
print(f"Average reward of the expert: {mean_reward}, {std_reward}.")

# Sample transitions from the expert policy
transitions = sample_expert_transitions(expert)
print(f"Number of transitions in demonstrations: {transitions.obs.shape[0]}.")

# Define generator algorithm
gen_algo = PPO(
    policy="MlpPolicy",
    env=env,
    seed=0,
    batch_size=min_batch_size,
    ent_coef=arglist.ent_coef,
    learning_rate=arglist.lr,
    gamma=arglist.discount,
    n_epochs=5,
    n_steps=min_batch_size,
    device=device,
)

# Define reward network
rwd_net = BasicRewardNet(env.unwrapped.envs[0].unwrapped.observation_space, env.unwrapped.envs[0].unwrapped.action_space)

# Create GAIL trainer
gail_trainer = gail.GAIL(
    demonstrations=transitions,
    venv=env,
    gen_algo=gen_algo,
    demo_batch_size=16,  # Batch size for discriminator
    reward_net=rwd_net,
    allow_variable_horizon=True,  # Allow variable episode lengths
)

print("Starting reward learning with GAIL.")

# Define training parameters
total_timesteps = 10000


# Define callback for logging
def log_callback(round_idx: int):
    obs = torch.tensor(transitions.obs, device=device)
    acts = torch.tensor(transitions.acts, device=device)

    # KL divergence between expert and generator
    kl_div = compute_kl_divergence(expert, gen_algo.policy, obs, acts, device)
    mean_reward, _ = evaluate_policy(model=gen_algo.policy, env=env, n_eval_episodes=10)

    writer.add_scalar("Valid/distance", kl_div, round_idx)
    writer.add_scalar("Valid/reward", mean_reward, round_idx)


# Start training
gail_trainer.train(
    total_timesteps=total_timesteps,
    callback=log_callback,
)

# Close TensorBoard writer
writer.close()
