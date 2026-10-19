"""SB3 vector-environment collection and off-policy training lifecycle."""

from tqdm import tqdm
from abc import ABC, abstractmethod
import numpy as np
from collections import deque
import gymnasium as gym
from stable_baselines3.common.vec_env import VecEnv

from logger.logger import Logger
from memory.memory import ReplayBuffer
from utils.evaluate import evaluate_atari
from agent.atari_agent import AtariDQNAgent
from utils.result import Result


class OffPolicyAlgorithm(ABC):
    """Base class for off-policy RL algorithms."""

    def __init__(self,
                 training_envs: VecEnv,
                 testing_envs: VecEnv | None,
                 buffer: ReplayBuffer,
                 agent: AtariDQNAgent,
                 logger: Logger,
                 device,
                 save_pth: str,
                 args):
        """Initialize the component and retain its configuration."""
        super(OffPolicyAlgorithm,self).__init__()
        self.training_envs = training_envs
        self.testing_envs = testing_envs
        self.buffer = buffer
        self.agent = agent.to(device)
        self.device = device
        self.logger = logger



        self.interaction_step = 0
        self.gradient_step = 0

        self.total_epoch = args.total_epoch
        self.num_training_envs = training_envs.num_envs
        self.epsilon = 0.0
        if args.interact_per_epoch <= 0:
            raise ValueError("interact_per_epoch must be positive")
        self.num_testing_envs = args.env.num_testing_envs
        self.test_interval = args.test_interval
        self.interact_per_epoch = args.interact_per_epoch
        self.test_episodes = args.test_episodes
        self.train_action_deterministic = args.train_action_deterministic
        self.save_interval = args.save_interval
        self.train_log_interval = args.train_log_interval

        self.save_pth = save_pth
        self.args = args
        self.episode_reward_buffer = deque(maxlen=args.reward_buffer_size)
        self.envs_rewards = np.zeros((self.num_training_envs,))
        # Subclasses may skip expensive per-update scalar extraction when the
        # result will not be logged. Direct calls to _update_policy keep metrics
        # by default, so tests and one-off diagnostics are unchanged.
        self._collect_update_metrics = True



    def update(self, batch, start_train=None)-> Result:
        """Store collected transitions and optionally perform one policy update."""
        with Result("buffer") as result:
            self._update_buffer(batch)
        if start_train is None:
            start_train = self.start_train()
        if start_train:
            update_policy_log = self._update_policy()
            result.add(update_policy_log)
            return result
        return result


    @abstractmethod
    def _update_buffer(self, batch):
        """
        update the replay buffer with given selected batch of data
        :param batch: the batch of data
        """

    @abstractmethod
    def _update_policy(self):
        """
        do gradient update to the policy with a batch of data sampled from replay buffer
        """

    def random_choose_action(self):
        """You can use this function to implement epsilon-greedy exploration strategy"""
        return False

    def interact_with_envs(self):
        """
        Interact with the environments and collect data, then return the
        collected batch of data.
        """
        batch = {key: [] for key in ('states', 'actions', 'rewards', 'next_states', 'dones')}
        with Result("interact") as result:
            for _ in range(self.interact_per_epoch):
                self.agent.epsilon = self.epsilon
                actions = self.agent.select_action(
                    self.current_states, deterministic=self.train_action_deterministic)
                next_obs, rewards, dones, infos = self.training_envs.step(actions)
                successors = next_obs.copy()
                terminals = dones.copy()
                for i, info in enumerate(infos):
                    if dones[i]:
                        successors[i] = info['terminal_observation']
                        terminals[i] = not info.get('TimeLimit.truncated', False)
                    if 'episode' in info:
                        self.episode_reward_buffer.append(info['episode']['r'])
                for key, value in zip(batch, (self.current_states, actions, rewards, successors, terminals)):
                    batch[key].append(value.copy())
                self.current_states = next_obs
                self.interaction_step += self.num_training_envs
                self.update_epsilon()
        batch = {key: np.stack(value) for key, value in batch.items()}
        if self.episode_reward_buffer:
            result.add_metric('reward_mean', np.mean(self.episode_reward_buffer))
            result.add_metric('reward_std', np.std(self.episode_reward_buffer))
        return batch, result

    def update_epsilon(self):
        """Hook for algorithms that schedule exploration after interaction."""

    def _interval_due(self, interval, name):
        """Trigger once when a counter crosses an interval boundary; zero disables."""
        if interval <= 0:
            return False
        bucket = self.interaction_step // interval
        previous = getattr(self, name, 0)
        setattr(self, name, bucket)
        return bucket > previous

    def test_condition(self):
        """
        check if it's time to test the policy
        """
        return self._interval_due(self.test_interval, '_last_test')

    def save_condition(self):
        """Return whether a checkpoint interval has elapsed."""
        return self._interval_due(self.save_interval, '_last_save')

    def initialize(self):
        """Reset training environments and retain their batched observations."""
        self.current_states = self.training_envs.reset()

    def train_log_condition(self):
        """Return whether a logging interval has elapsed."""
        return self._interval_due(self.train_log_interval, '_last_log')

    def start_train(self):
        """Wait for warmup and enough replay transitions for a minibatch."""
        return self.interaction_step >= self.args.algorithm.get('learning_starts', 5000) and len(self.buffer) >= self.batch_size

    def run(self):
        """
        run the training loop of the algorithm, this is a typical procedure of offpolicy algorithm
        """

        self.initialize() # reset envs
        for epoch in tqdm(range(self.total_epoch), desc="Epoch", unit="epoch"):
            collected_batch, interact_result = self.interact_with_envs()
            # Decide logging before the update so costly device synchronizations
            # for scalar metrics happen only on logging boundaries.
            log_this_update = self.train_log_condition()
            self._collect_update_metrics = log_this_update
            train_result = self.update(collected_batch)
            if log_this_update:
                train_result.add(interact_result)
                self.logger.log_train(epoch, self.interaction_step, self.gradient_step, train_result)


            if self.test_condition():
                test_result = evaluate_atari(self.agent, self.test_episodes,self.args.env, seed=self.args.seed)
                self.logger.log_test(epoch, self.interaction_step, self.gradient_step, test_result)

            if self.save_condition():
                self.agent.save(self.save_pth)





        self.agent.save(self.save_pth)
