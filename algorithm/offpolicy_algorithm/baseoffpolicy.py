from tqdm import tqdm
from abc import ABC, abstractmethod
import numpy as np
from collections import deque
import gymnasium as gym
from typing import Dict, Any, cast
from gymnasium.vector import VectorEnv

from logger.logger import Logger
from memory.memory import ReplayBuffer
from utils.evaluate import evaluate_atari
from agent.atari_agent import AtariDQNAgent
from utils.result import Result


class OffPolicyAlgorithm(ABC):
    """Base class for off-policy RL algorithms."""

    def __init__(self, 
                 training_envs: VectorEnv, 
                 testing_envs: gym.Env, 
                 buffer: ReplayBuffer, 
                 agent: AtariDQNAgent, 
                 logger: Logger, 
                 device, 
                 save_pth: str, 
                 args):
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
        self.num_training_envs = args.env.num_training_envs
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

    
    
    def update(self, batch, start_train)-> Result:
        with Result("buffer") as result:
            self._update_buffer(batch)
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
        interact_steps_per_env = self.interact_per_epoch

        # Explicit annotation so Pylance knows the values can become arrays
        batch: Dict[str, Any] = dict(
            states=[], actions=[], rewards=[], next_states=[], dones=[]
        )

        with Result("interact") as result:
            for _ in range(interact_steps_per_env):
                # --- 1. Select actions for all envs (vectorized) ---
                actions = []
                for i in range(self.num_training_envs):
                    # Assume self.current_states[i] is the latest state for env i
                    action = self.agent.select_action(
                        self.current_states[i], deterministic=False
                    )
                    actions.append(action)

                # --- 2. Step all envs simultaneously (vectorized API) ---
                actions = np.array(actions)  # shape (num_envs,)
                next_obs, rewards, terminated, truncated, infos = \
                    self.training_envs.step(actions)
                infos = cast(list[dict], infos)

                # --- 3. Store transitions in the batch ---
                for i in range(self.num_training_envs):
                    done = terminated or truncated
                    batch["states"].append(self.current_states[i])
                    batch["actions"].append(actions[i])
                    batch["rewards"].append(rewards[i])
                    batch["next_states"].append(next_obs[i])
                    batch["dones"].append(done)

                # --- 4. Handle episode terminations ---
                for i in range(self.num_training_envs):
                    info = infos[i]
                    if "episode" in info:
                        self.episode_reward_buffer.append(info["episode"]["r"])

                # --- 5. Reset terminated envs and update current states ---
                # For a vectorized env, gymnasium usually auto-resets terminated
                # sub-envs if you wrap it with `gym.vector.SyncVectorEnv`. If not,
                # you'll need to manually reset them here.
                self.current_states = next_obs

            # Convert lists to numpy arrays
            for k, v in batch.items():
                batch[k] = np.stack(v, axis=0)

        if len(self.episode_reward_buffer) > 0:
            result.add_metric("reward_mean", np.mean(self.episode_reward_buffer))
            result.add_metric("reward_std", np.std(self.episode_reward_buffer))

        return batch, result

    def test_condition(self):
        """
        check if it's time to test the policy
        """
        return self.interaction_step % self.test_interval == 0

    def save_condition(self):
        """check if it's time to save the checkpoint"""
        if self.interaction_step % self.save_interval == 0:
            return True 
        return False

    def initialize(self):
        self.observations = self.training_envs.reset()
    
    def train_log_condition(self):
        if self.interaction_step%self.train_log_interval==0:
            return True 
        return False

    def start_train(self):
        return self.interaction_step>=5000

    def run(self):
        """
        run the training loop of the algorithm, this is a typical procedure of offpolicy algorithm
        """
        
        self.initialize() # reset envs
        for epoch in tqdm(range(self.total_epoch), desc="Epoch", unit="epoch"):
            if self.test_condition():
                test_result = evaluate_atari(self.agent, self.test_episodes,self.args.env)
                self.logger.log_test(epoch, self.interaction_step, self.gradient_step, test_result)
            
            collected_batch, interact_result = self.interact_with_envs()
            train_result = self.update(collected_batch, self.start_train())
            if self.train_log_condition():
                train_result.add(interact_result)
                self.logger.log_train(epoch, self.interaction_step, self.gradient_step, train_result)
                
            
            if self.save_condition():
                self.agent.save(self.save_pth)




