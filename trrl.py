"""Trust Region Reward Learning (TRRL).

Trains a reward function whose induced policy is monotonically improved towards the expert policy.
"""
import os
import time
from typing import Callable, Iterator, Mapping, Optional, Type, cast
import copy
import tqdm
import torch
import numpy as np
import gymnasium as gym
from functools import wraps
from stable_baselines3.common.evaluation import evaluate_policy
import datetime
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage


from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common import policies, vec_env, evaluation, preprocessing

from imitation.algorithms import base as algo_base
from imitation.algorithms import base
from imitation.data import types
from imitation.util import logger, networks, util
from imitation.util.util import make_vec_env
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper

from reward_function import RwdFromRwdNet, RewardNet
import rollouts
import random

import torch.utils.tensorboard as tb


def timeit_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        temp_str = str(func.__code__)
        # print(f"Function {temp_str[-30:]} {func.__name__}  executed in {end_time - start_time} seconds")
        return result

    return wrapper


class CustomResetEnv(gym.Env):
    def __init__(self, env):
        self.env = env

    def reset(self, custom_state=None):
        obs = self.env.reset()
        if custom_state is not None:
            self.env.state = custom_state  # 设置自定义状态
            obs = self.env._get_obs()  # 调用环境的观测生成函数
        return obs

    def step(self, action):
        return self.env.step(action)


class TRRL(algo_base.DemonstrationAlgorithm[types.Transitions]):
    """Trust Region Reward Learning (TRRL).

        Trains a reward function whose induced policy is monotonically improved towards the expert policy.
    """

    def __init__(
            self,
            *,
            venv: vec_env.VecEnv,
            expert_policy: policies = None,
            demonstrations: base.AnyTransitions,
            demo_batch_size: int,
            custom_logger: Optional[logger.HierarchicalLogger] = None,
            reward_net: RewardNet,
            discount: np.float32,
            t_kl:np.float32,
            avg_diff_coef: np.float32,
            l2_norm_coef: np.float32,
            l2_norm_upper_bound: np.float32,
            ent_coef: np.float32 = 0.01,
            rwd_opt_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
            device: torch.device = torch.device("cpu"),
            log_dir: types.AnyPath = "output/",
            allow_variable_horizon: bool = False,
            n_policy_updates_per_round=100_000,
            n_reward_updates_per_round=10,
            n_episodes_adv_fn_est=32,
            n_timesteps_adv_fn_est=64,
            **kwargs,
    ):
        """
        Builds TRRL.

        :param venv: The vectorized environment to train on.
        :param expert_policy: The expert polocy in the form of stablebaseline3 policies. This is used to
            calculate the difference between the expert policy and the learned policy.
        :param demonstrations: Demonstrations to use for training. The input demo should be flatened.
        :param old_policy: The policy model to use for the old policy (Stable Baseline 3).
        :param demo_batch_size: The number of samples in each batch of expert data.
        :param custom_logger: Where to log to; if None (default), creates a new logger.
        :param reward_net: reward network.
        :param discount: discount factor. A value between 0 and 1.
        :param avg_diff_coef: coefficient for `r_old - r_new`.
        :param l2_norm_coef: coefficient for the max difference between r_new and r_old.
            In the practical algorithm, the max difference is replaced
            by an average distance for the differentiability.
        :param l2_norm_upper_bound: Upper bound for the l2 norm of the difference between current and old reward net
        :param ent_coef: coefficient for policy entropy.
        :param rwd_opt_cls: The optimizer for reward training
        :param n_policy_updates_per_round: The number of rounds for updating the policy per global round.
        :param n_reward_updates_per_round: The number of rounds for updating the reward per global round.
        :param n_episodes_adv_fn_est: Number of episodes for advantage function estimation.
        :param n_timesteps_adv_fn_est: number of timesteps for advantage function estimation.
        :param log_dir: Directory to store TensorBoard logs, plots, etc. in.
        :param kwargs: Keyword arguments to pass to the RL algorithm constructor.

        :raises: ValueError: if `dqn_kwargs` includes a key
                `replay_buffer_class` or `replay_buffer_kwargs`.
        """
        self._rwd_opt_cls = rwd_opt_cls
        self._old_policy = None
        self._old_reward_net = None
        self.ent_coef = ent_coef
        self.avg_diff_coef = avg_diff_coef
        self.l2_norm_coef = l2_norm_coef
        self.l2_norm_upper_bound = l2_norm_upper_bound
        self.t_kl = t_kl
        # self.expert_state_action_density = self.est_expert_demo_state_action_density(demonstrations)
        self.venv = venv
        self.device = device
        self._expert_policy = expert_policy
        self.demonstrations = demonstrations
        self.demo_batch_size = demo_batch_size
        super().__init__(
            demonstrations=demonstrations,
            custom_logger=custom_logger,
            allow_variable_horizon=allow_variable_horizon,
        )
        self.venv = venv
        self._new_policy = None  # Initialize _new_policy
        self._reward_net = reward_net.to(device)
        self._rwd_opt = self._rwd_opt_cls(self._reward_net.parameters(), lr=0.0005)
        self.discount = discount
        self.n_policy_updates_per_round = n_policy_updates_per_round
        self.n_reward_updates_per_round = n_reward_updates_per_round
        self.n_episodes_adv_fn_est = n_episodes_adv_fn_est
        self.n_timesteps_adv_fn_est = n_timesteps_adv_fn_est
        self._log_dir = util.parse_path(log_dir)
        # self.logger = logger.configure(self._log_dir)
        self._global_step = 0
        self.MAX_BUFFER_SIZE = 1000  # 定义缓冲区最大值
        self.trajectory_buffer = []  # 初始化缓冲区
        self.current_iteration = 0  # 当前策略迭代次数
        self.recent_policy_window = 5  # 只从最近5个策略生成的轨迹中采样

    def store_trajectory(self, trajectory):
        """存储轨迹，并将它与当前策略的迭代次数相关联"""
        if len(self.trajectory_buffer) >= self.MAX_BUFFER_SIZE:
            self.trajectory_buffer.pop(0)  # 移除最早的轨迹
        self.trajectory_buffer.append((trajectory, self.current_iteration))  # 存储轨迹和对应的策略迭代次数

    def sample_old_trajectory(self):
        """只从最近策略生成的轨迹中进行采样"""
        recent_trajectories = [
            traj for traj, iteration in self.trajectory_buffer
            if iteration >= self.current_iteration - self.recent_policy_window
        ]
        if len(recent_trajectories) == 0:
            raise ValueError("没有足够的最近轨迹可供采样")
        return random.choice(recent_trajectories)

    @property
    @timeit_decorator
    def expert_kl(self) -> float:
        """KL divergence between the expert and the current policy.
        A Stablebaseline3-format expert policy is required.

        :return: The average KL divergence between the the expert policy and the current policy
        """
        assert self._old_policy is not None
        assert isinstance(self._old_policy.policy, policies.ActorCriticPolicy)
        assert isinstance(self._expert_policy.policy, policies.ActorCriticPolicy)
        obs = copy.deepcopy(self.demonstrations.obs)
        acts = copy.deepcopy(self.demonstrations.acts)

        obs_th = torch.as_tensor(obs, device=self.device)
        acts_th = torch.as_tensor(acts, device=self.device)

        # 确保模型的权重在同一设备上
        self._old_policy.policy.to(self.device)
        self._expert_policy.policy.to(self.device)
        # print(obs_th.shape)
        # print(acts_th.shape)

        input_values, input_log_prob, input_entropy = self._old_policy.policy.evaluate_actions(obs_th, acts_th)
        target_values, target_log_prob, target_entropy = self._expert_policy.policy.evaluate_actions(obs_th, acts_th)

        kl_div = torch.mean(torch.dot(torch.exp(target_log_prob), target_log_prob - input_log_prob))

        return (float(kl_div))

    @property
    def evaluate_policy(self) -> float:
        """Evalute the true expected return of the learned policy under the original environment.

        :return: The true expected return of the learning policy.
        """
        assert self._old_policy is not None
        assert isinstance(self._old_policy.policy, policies.ActorCriticPolicy)

        mean_reward, std_reward = evaluation.evaluate_policy(model=self._old_policy, env=self.venv)
        return mean_reward

    def log_saving(self) -> None:
        """Save logs containing the following info:
                1. KL divergence between the expert and the current policy;
                2. Evaluations of the current policy.
        """
        # TODO 
        pass

    def set_demonstrations(self, demonstrations) -> None:
        self.demonstrations = demonstrations

    def reset(self, reward_net: RewardNet = None):
        """Reset the reward network and the iteration counter.
        
        Args:
            reward_net: The reward network to set as.
        """
        self._reward_net = reward_net
        self._old_reward_net = None

    def est_expert_demo_state_action_density(self, demonstration: base.AnyTransitions) -> np.ndarray:
        pass

    #############################################################################################
    def compute_is_weights(self, old_policy, new_policy, observations, actions):
        """
        Compute the importance sampling (IS) weights.

        Args:
            old_policy: The old policy used to generate the original trajectories.
            new_policy: The new policy that we are trying to evaluate.
            observations: The observations from the trajectory.
            actions: The actions taken in the trajectory.

        Returns:
            weights: The computed IS weights.
        """
        observations = torch.as_tensor(observations, device=self.device)
        actions = torch.as_tensor(actions, device=self.device)

        # 确保模型的权重在同一设备上
        self._old_policy.policy.to(self.device)
        self._expert_policy.policy.to(self.device)

        ############################test

        # 打印形状
        # print("Observations shape:", observations.shape)
        # print("Actions shape:", actions.shape)

        # observations = observations.to(self.device)
        # actions = actions.to(self.device)
        # old_prob = old_policy.policy.get_distribution(observations).log_prob(actions)
        # new_prob = new_policy.policy.get_distribution(observations).log_prob(actions)

        old_prob = old_policy.policy.evaluate_actions(observations, actions)[1]
        new_prob = new_policy.policy.evaluate_actions(observations, actions)[1]

        # print("Old policy output shape:", old_prob.shape)
        # print("New policy output shape:", new_prob.shape)

        weights = torch.exp(new_prob - old_prob)
        return weights

    ########################################################################

    # @timeit_decorator
    def est_adv_fn_old_policy_cur_reward(self, starting_state: np.ndarray, starting_action: np.ndarray,
                                         n_timesteps: int, n_episodes: int, use_mc=False) -> torch.Tensor:
        """Use Monte-Carlo or Importance Sampling to estimate the advantage function of the given state and action under the
        old policy and the current reward network

        Advantage function: A^{\pi_{old}}_{r_\theta}(s,a) = Q^{\pi_{old}}_{r_\theta}(s,a) - V^{\pi_{old}}_{r_\theta}(s,a)

        Args:
            starting_state: The state to estimate the advantage function for.
            starting_action: The action to estimate the advantage function for.
            n_timesteps: The length of a rollout.
            n_episodes: The number of simulated rollouts.
            use_mc: Boolean flag to determine whether to use Monte Carlo.

        Returns:
            the estimated value of advantage function at `starting_state` and `starting_action`
        """
        rng = np.random.default_rng(0)

        # TODO 这里用的是不带新Reward的env?
        env = VecTransposeImage(make_vec_env(
            env_name=self.venv.unwrapped.envs[0].unwrapped.spec.id,
            n_envs=self.venv.num_envs,
            rng=rng,
        )
        )

        if isinstance(self.venv.unwrapped.envs[0].unwrapped.action_space, gym.spaces.Discrete):
            starting_a = starting_action.astype(int)
        else:
            starting_a = starting_action
        if isinstance(self.venv.unwrapped.envs[0].unwrapped.observation_space, gym.spaces.Discrete):
            starting_s = starting_state.astype(int)
        else:
            starting_s = starting_state

        q = torch.zeros(1).to(self.device)

        for ep_idx in range(n_episodes):
            if use_mc:
                # Monte Carlo: Sample a new trajectory
                tran = rollouts.generate_transitions(
                    self._old_policy,
                    env,
                    rng=rng,
                    n_timesteps=n_timesteps,
                    starting_state=starting_s,
                    starting_action=starting_a,
                    truncate=True,
                )
                # tran = rollout.generate_transitions(
                #     self._old_policy,
                #     env,
                #     rng=rng,
                #     n_timesteps=n_timesteps,
                #     #starting_state=starting_s,
                #     #starting_action=starting_a,
                #     truncate=True,
                # )
                self.store_trajectory(tran)  # Store the new trajectory in buffer
            else:
                # Importance Sampling: Sample an old trajectory from the buffer
                tran = self.sample_old_trajectory()
                # After applying IS, treat the new weighted trajectory as a new one and store it
                self.store_trajectory(tran)  # 将经过 IS 处理后的轨迹重新加入池子
            state_th, action_th, next_state_th, done_th = self._reward_net.preprocess(tran.obs, tran.acts,
                                                                                      tran.next_obs, tran.dones)

            state_th, action_th, next_state_th, done_th = self._reward_net.preprocess(tran.obs, tran.acts,
                                                                                      tran.next_obs, tran.dones)
            rwds = self._reward_net(state_th, action_th, next_state_th, done_th)
            discounts = torch.pow(torch.ones(n_timesteps, device=self.device) * self.discount,
                                  torch.arange(0, n_timesteps, device=self.device))

            if use_mc:
                # Use Monte Carlo estimation
                q += torch.dot(rwds, discounts)
            else:
                # Use Importance Sampling estimation
                weights = self.compute_is_weights(self._old_policy, self._new_policy, tran.obs, tran.acts)
                q += torch.dot(weights * rwds, discounts)

        # The v calculation remains unchanged
        v = torch.zeros(1).to(self.device)
        '''
        if isinstance(self.venv.unwrapped.envs[0].unwrapped.action_space, gym.spaces.Discrete):
            # if the action space is discrete, then V(s,a) can be calculated as the expectation of Q(s,a) over all a's 
            state_th = util.safe_to_tensor(starting_state).to(self.device)
            state_th = cast(
                torch.Tensor,
                preprocessing.preprocess_obs(
                    state_th,
                    self.venv.unwrapped.envs[0].unwrapped.observation_space,
                    True,
                ),
            )
            with torch.no_grad():
                self._old_policy.policy.forward(obs=torch.as_tensor(state_th, device=self.device))
                PPO.policy.for
            pass
            # if the action space is fully or partially continuous, then V(s,a) is approximated by Monte Carlo simulation.
        '''
        j = 0
        for ep_idx in range(n_episodes):
            tran = rollouts.generate_transitions(
                self._old_policy,
                env,
                n_timesteps=n_timesteps,
                rng=rng,
                starting_state=starting_s,
                starting_action=None,
                truncate=True,
            )
            #
            # tran = rollout.generate_transitions(
            #     self._old_policy,
            #     env,
            #     n_timesteps=n_timesteps,
            #     rng=rng,
            #     #starting_state=starting_s,
            #     #starting_action=None,
            #     truncate=True,
            # )
            state_th, action_th, next_state_th, done_th = self._reward_net.preprocess(tran.obs, tran.acts,
                                                                                      tran.next_obs, tran.dones)

            # Ensure tensors are on the correct device
            state_th = state_th.to(self.device)
            action_th = action_th.to(self.device)
            next_state_th = next_state_th.to(self.device)
            done_th = done_th.to(self.device)

            rwds = self._reward_net(state_th, action_th, next_state_th, done_th)
            discounts = torch.pow(torch.ones(n_timesteps, device=self.device) * self.discount,
                                  torch.arange(0, n_timesteps, device=self.device))

            if use_mc:
                v += torch.dot(rwds, discounts)
            else:
                weights = self.compute_is_weights(self._old_policy, self._new_policy, tran.obs, tran.acts)
                v += torch.dot(weights * rwds, discounts)

        env.close()
        return (q - v) / n_episodes

    @timeit_decorator
    def train_new_policy_for_new_reward(self) -> policies.BasePolicy:
        """Update the policy to maximise the rewards under the new reward function. The PPO algorithm will be used.

        Returns:
            A gym PPO policy optimised for the current reward network
        """
        rng = np.random.default_rng(0)
        venv = VecTransposeImage(make_vec_env(
            env_name=self.venv.unwrapped.envs[0].unwrapped.spec.id,
            n_envs=self.venv.num_envs,
            rng=rng,
        )
        )

        # setup an env with the reward being the current reward network
        rwd_fn = RwdFromRwdNet(rwd_net=self._reward_net)
        venv_with_cur_rwd_net = RewardVecEnvWrapper(
            venv=venv,
            reward_fn=rwd_fn
        )

        _ = venv_with_cur_rwd_net.reset()

        new_policy = PPO(
            policy=MlpPolicy,
            env=venv_with_cur_rwd_net,
            batch_size=64,
            ent_coef=self.ent_coef,
            learning_rate=0.0005,
            n_epochs=10,
            n_steps=64,
            gamma=self.discount
        )

        new_policy.learn(self.n_policy_updates_per_round)

        venv_with_cur_rwd_net.close()
        venv.close()

        # Store the new policy
        self._new_policy = new_policy

        # 在策略训练完成后，更新策略迭代次数
        self.current_iteration += 1

        return new_policy

    @timeit_decorator
    def update_reward(self, use_mc=False):
        """Perform a single reward update by conducting a complete pass over the demonstrations, 
        optionally using provided samples. The loss is adapted from the constrained optimisation 
        problem of the trust region reward learning by Lagrangian multipliers (moving the constraints 
        into the objective function).
        
        Args:
            use_mc: Boolean flag to determine whether to use Monte Carlo for advantage function estimation.
            cur_round: The number of current round of reward-policy iteration
        Returns:
            The updated reward network
        """
        # TODO: consider optimise a reward network from scratch
        # initialise the optimiser for the reward net
        # Do a complete pass on the demonstrations, i.e., sampling sufficient batches for performing GD.
        batch_iter = self._make_reward_train_batches()

        for batch in batch_iter:

            start_batch = time.time()
            # estimate the advantage function
            obs = batch["state"]
            acts = batch["action"]
            next_obs = batch["next_state"]
            dones = batch["done"]

            loss = torch.zeros(1).to(self.device)

            # estimated average estimated advantage function values
            cumul_advantages = torch.zeros(1).to(self.device)

            for idx in range(obs.shape[0]):
                cumul_advantages += self.est_adv_fn_old_policy_cur_reward(starting_state=obs[idx],
                                                                          starting_action=acts[idx],
                                                                          n_timesteps=self.n_timesteps_adv_fn_est,
                                                                          n_episodes=self.n_episodes_adv_fn_est,
                                                                          use_mc=use_mc)

            avg_advantages = cumul_advantages / obs.shape[0]
            state_th, action_th, next_state_th, done_th = self._reward_net.preprocess(obs, acts, next_obs, dones)

            if self._old_reward_net is None:
                reward_diff = self._reward_net(state_th, action_th, next_state_th, done_th) - torch.ones(1).to(
                    self.device)
                print("self._old_reward_net is None")
            else:
                # use `predict_th` for `self._old_reward_net` as its gradient is not needed
                # TODO: 第一轮迭代，diff=0，因为old和new RewardNet存的相同
                reward_diff = self._reward_net(state_th, action_th, next_state_th,
                                               done_th) - self._old_reward_net.predict_th(obs, acts, next_obs,
                                                                                          dones).to(self.device)

            # print("reward_diff:",reward_diff)

            # TODO: two penalties should be calculated over all state-action pairs
            # 在所有的S-A上计算； S-A去重
            avg_reward_diff = torch.mean(reward_diff)
            l2_norm_reward_diff = torch.norm(reward_diff, p=2)

            ####################################################################
            #adaptive Lagrange multiplier
            distance = self.expert_kl
            t_kl= self.t_kl
            print("\nThreshold is:", t_kl,"coef is:",self.avg_diff_coef)
            if distance > self.t_kl * 1.5:
                self.avg_diff_coef *= 1.2  # 增加 Lagrange 乘子系数
            elif distance < self.t_kl / 1.5:
                self.avg_diff_coef /= 1.2  # 减小 Lagrange 乘子系数

            # Ensure avg_diff_coef is a tensor before applying clamp
            self.avg_diff_coef = torch.tensor(self.avg_diff_coef)
            self.avg_diff_coef = torch.clamp(self.avg_diff_coef, min=1e-5, max=1)

            # 设定参考阈值与缩放因子（根据实际情况调整）
            l2_norm_target_lower = 0.01     # 当l2_norm_reward_diff低于该值时，说明更新过于保守，可减少约束
            l2_norm_target_upper = 1     # 当l2_norm_reward_diff高于该值时，说明更新过大，需要加强约束
            scaling_factor_increase = 1.1   # 增大系数时的倍率
            scaling_factor_decrease = 1.1   # 减小系数时的倍率

            if l2_norm_reward_diff > l2_norm_target_upper:
                # 更新幅度过大，需要加强约束
                self.l2_norm_coef *= scaling_factor_increase
            elif l2_norm_reward_diff < l2_norm_target_lower:
                # 更新幅度过小，可能过于保守，适度放宽约束
                self.l2_norm_coef /= scaling_factor_decrease

            # 限制 l2_norm_coef 的范围，防止出现过大或过小的情况
            self.l2_norm_coef = torch.tensor(self.l2_norm_coef)
            self.l2_norm_coef = torch.clamp(self.l2_norm_coef, min=1e-5, max=1)

            # 打印或记录当前状态，便于调试和观察
            print(f"Current L2 diff: {l2_norm_reward_diff.item():.4f}, adjusted l2_norm_coef: {self.l2_norm_coef.item():.6f}")

            ####################################################################
            loss = avg_advantages + self.avg_diff_coef * avg_reward_diff - self.l2_norm_coef * l2_norm_reward_diff + self.l2_norm_upper_bound
            print(self._global_step, "loss:", loss, "avg_advantages:", avg_advantages, "avg_reward_diff:",
                  avg_reward_diff, "l2_norm_reward_diff:", l2_norm_reward_diff)

            loss = - loss * (self.demo_batch_size / self.demonstrations.obs.shape[0])

            self._rwd_opt.zero_grad()
            loss.backward()
            self._rwd_opt.step()

            writer.add_scalar("Train/loss", loss.item(), self._global_step)
            writer.add_scalar("Train/avg_advantages", avg_advantages.item(), self._global_step)
            writer.add_scalar("Train/avg_reward_diff", avg_reward_diff.item(), self._global_step)

            end_batch = time.time()
            print("batch time:", end_batch - start_batch)

            self._global_step += 1

    # @timeit_decorator
    def train(self, n_rounds: int, callback: Optional[Callable[[int], None]] = None):
        """
        Args:
            n_rounds: An upper bound on the iterations of training.
            callback: A function called at the end of every round which takes in a
                single argument, the round number.
        """
        # TODO: Make the initial reward net oupput <= 1 
        # Iteratively train a reward function and the induced policy.
        global writer
        # 获取当前时间并格式化
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # 创建带时间戳的日志目录，例如：logs/20231201_103000
        log_dir = f"logs/{current_time}"

        writer = tb.SummaryWriter(log_dir=log_dir, flush_secs=1)
        #writer = tb.SummaryWriter('./logs/', flush_secs=1)

        print("n_policy_updates_per_round:", self.n_policy_updates_per_round)
        print("n_reward_updates_per_round:", self.n_reward_updates_per_round)

        # 设置保存模型的轮数间隔
        save_interval = 50  

        for r in tqdm.tqdm(range(0, n_rounds), desc="round"):
            # Update the policy as the one optimal for the updated reward.

            self._old_policy = self.train_new_policy_for_new_reward()

            # Determine whether to use Monte Carlo or Importance Sampling
            use_mc = (r % 200 == 0)

            # Update the reward network.
            for _ in range(self.n_reward_updates_per_round):
                #print("reward_updates_per_round is",self.n_reward_updates_per_round)################################
                self.update_reward(use_mc=use_mc)

            self._old_reward_net = copy.deepcopy(self._reward_net)
            distance = self.expert_kl
            reward = self.evaluate_policy

            writer.add_scalar("Valid/distance", distance, r)
            writer.add_scalar("Valid/reward", reward, r)

            self.logger.record("round " + str(r),
                               'Distance: ' + str(distance) + '. Reward: ' + str(self.evaluate_policy))
            self.logger.dump(step=10)

            # 每隔 save_interval 轮保存一次模型参数
            if (r + 1) % save_interval == 0:
                # 使用state_dict保存模型参数
                save_path = os.path.join(log_dir, f"reward_net_state_dict_round_{r+1}.pth")
                torch.save(self._reward_net.state_dict(), save_path)
                print(f"Saved reward net state dict at {save_path}")

            if callback:
                callback(r)

        writer.close()

    @property
    def policy(self) -> policies.BasePolicy:
        return self._old_policy

    @property
    def reward_net(self) -> RewardNet:
        return self._reward_net

    def _make_reward_train_batches(
            self,
    ) -> Iterator[Mapping[str, torch.Tensor]]:
        """Build and return training batches for the reward update.

        Args:
            expert_samples: Same as expert demonstrations.

        Returns:
            The training batch: state, action, next state, dones.
        """

        for start in range(0, self.demonstrations.obs.shape[0], self.demo_batch_size):
            end = start + self.demo_batch_size
            obs = self.demonstrations.obs[start:end]
            acts = self.demonstrations.acts[start:end]
            next_obs = self.demonstrations.next_obs[start:end]
            dones = self.demonstrations.dones[start:end]

            batch_dict = {
                "state": obs,
                "action": acts,
                "next_state": next_obs,
                "done": dones,
            }

            yield batch_dict
