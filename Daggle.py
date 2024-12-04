"""
This is the runner of using DAgger (Dataset Aggregation) to infer the optimal policy.
"""
import os
import arguments
import torch
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.util.util import make_vec_env
from stable_baselines3 import PPO
from imitation.algorithms.bc import BC
import torch.utils.tensorboard as tb
import rollouts
import logging
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

class NullLogger(logging.Logger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def handle(self, record):
        # 忽略所有日志记录
        pass

# 创建空日志记录器实例

# Parse arguments
arglist = arguments.parse_args()
rng = np.random.default_rng(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
)
)
print(f"Environment set to: {arglist.env_name}")

# Initialize TensorBoard logger
writer = tb.SummaryWriter(log_dir="logs/DAgger", flush_secs=1)

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
        starting_state= None, #np.array([0.1, 0.1, 0.1, 0.1]),
        starting_action=None, #np.array([[1,], [1,],], dtype=np.integer)
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

# Evaluate expert policy
mean_reward, std_reward = evaluate_policy(model=expert, env=env)
print(f"Average reward of the expert: {mean_reward}, {std_reward}.")

bc_trainer = BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    batch_size=min_batch_size,  # Batch size for supervised learning
    optimizer_kwargs={"lr": 3e-4},  # Learning rate
    device=device,
    rng=rng
)

# Initialize DAgger trainer
dagger_trainer = SimpleDAggerTrainer(
    venv=env,
    expert_policy=expert.policy,
    scratch_dir='scratch_dir',  # 指定一个有效的目录路径
    beta_schedule=None,  # You can define a beta schedule if needed
    #batch_size=min_batch_size,
    bc_trainer=bc_trainer,
    rng=rng,
    #device=device,
)

print("Starting DAgger training.")

# Define training parameters
total_timesteps = 10000  # Number of timesteps for training

# Define a class to manage logging context for DAgger training
class DAggerLogger:
    def __init__(self, writer, expert, dagger_trainer, env, device):
        self.writer = writer
        self.expert = expert
        self.dagger_trainer = dagger_trainer
        self.env = env
        self.device = device
        self.step_idx = 0  # Initialize step index

    def log_metrics(self):
        """Log metrics during training."""
        # Evaluate the policy
        obs = torch.tensor(self.transitions.obs, device=self.device)
        acts = torch.tensor(self.transitions.acts, device=self.device)

        # Compute KL divergence and evaluate reward
        kl_div = compute_kl_divergence(self.expert, self.bc_trainer.policy, obs, acts, self.device)
        mean_reward, _ = evaluate_policy(model=self.dagger_trainer.policy, env=self.env, n_eval_episodes=10)

        # Log to TensorBoard
        self.writer.add_scalar("Valid/reward", mean_reward, self.step_idx)
        self.writer.add_scalar("Valid/distance", kl_div, self.epoch_idx)

        # Print log
        print(f"Epoch {self.epoch_idx}: KL Divergence = {kl_div:.4f}, Reward = {mean_reward:.4f}")
        self.step_idx += 1  # Increment step index

# Instantiate the logger
dagger_logger = DAggerLogger(writer, expert, dagger_trainer, env, device)

# Training loop with logging
total_steps = 100_000
for step in range(0, total_steps, 10_000):  # Adjust step size as needed
    dagger_trainer.train(total_timesteps=10_000)
    dagger_logger.log_metrics()

# Close TensorBoard writer
writer.close()
