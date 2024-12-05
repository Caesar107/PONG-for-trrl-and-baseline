"""
This is the runner of using BC (Behavioral Cloning) to infer the optimal policy.
"""
import os
import arguments
import torch
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.algorithms.bc import BC
from imitation.util.util import make_vec_env
from stable_baselines3 import PPO
import torch.utils.tensorboard as tb
import rollouts
import logging
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

# Remove default terminal logging
logger = logging.getLogger()
logger.handlers = []

# Parse arguments
arglist = arguments.parse_args()
rng = np.random.default_rng(0)

# Force device to CPU (this avoids using GPU even if CUDA is available)
device = torch.device("cpu")

min_batch_size = 16

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

    # Compute KL divergence
    kl_divergence = torch.mean(torch.dot(torch.exp(target_log_prob), target_log_prob - input_log_prob)).item()

    return kl_divergence

# Create environment
env = VecTransposeImage(make_vec_env(
    arglist.env_name,
    n_envs=1,  # Parallel environments
    rng=rng,
))
print(f"Environment set to: {arglist.env_name}")

# Initialize TensorBoard logger
writer = tb.SummaryWriter(log_dir="logs/BC", flush_secs=1)

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
        n_epochs=1,
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

# Create BC trainer
bc_trainer = BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    batch_size=min_batch_size,  # Batch size for supervised learning
    optimizer_kwargs={"lr": 3e-4},  # Learning rate
    device=device,
    rng=rng
)

print("Starting Behavioral Cloning training.")

# Define total training timesteps
total_timesteps = 10000  # Same as AIRL

# Modify the BCLogger class to use total_timesteps-based rounds
class BCLogger:
    def __init__(self, writer, expert, bc_trainer, env, transitions, device):
        self.writer = writer
        self.expert = expert
        self.bc_trainer = bc_trainer
        self.env = env
        self.transitions = transitions
        self.device = device
        self.timestep_idx = 0  # Initialize timestep index

    def on_epoch_end(self):
        """Log metrics at the end of each epoch."""
        obs = torch.tensor(self.transitions.obs, device=self.device)
        acts = torch.tensor(self.transitions.acts, device=self.device)

        # Compute KL divergence and evaluate reward
        kl_div = compute_kl_divergence(self.expert, self.bc_trainer.policy, obs, acts, self.device)
        mean_reward, _ = evaluate_policy(model=self.bc_trainer.policy, env=self.env, n_eval_episodes=10)

        # Log to TensorBoard using timestep_idx
        self.writer.add_scalar("Valid/distance", kl_div, self.timestep_idx)
        self.writer.add_scalar("Valid/reward", mean_reward, self.timestep_idx)

        # Print log
        print(f"Timestep {self.timestep_idx}: KL Divergence = {kl_div:.4f}, Reward = {mean_reward:.4f}")

        # Increment timestep index (based on training progress)
        self.timestep_idx += total_timesteps // 625  # Adjust according to number of epochs

# Instantiate the logger with total timesteps logic
bc_logger = BCLogger(writer, expert, bc_trainer, env, transitions, device)

# Train BC with adjusted logger
bc_trainer.train(
    n_epochs=625,  # Number of epochs to train
    on_epoch_end=bc_logger.on_epoch_end,  # Pass the modified logging method
)



