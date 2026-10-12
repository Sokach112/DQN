from .baseoffpolicy import OffPolicyAlgorithm
from memory.memory import ReplayBuffer
import torch
import torch.nn.functional as F
import numpy as np
from utils.result import Result
from logger.logger import Logger
from agent.atari_agent import AtariDQNAgent


OPTIMIZER_DICT = {
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
    "RMSprop": torch.optim.RMSprop,
    "AdamW": torch.optim.AdamW,
}

LOSS_DICT = {
    "mse": torch.nn.MSELoss(),
    "huber": torch.nn.SmoothL1Loss(),
}


class DQN(OffPolicyAlgorithm):
    """DQN algorithm implementation."""

    def __init__(self, training_envs, testing_envs, buffer: ReplayBuffer,
                 agent: AtariDQNAgent, logger: Logger, device, save_pth: str,
                 args, target_agent=None):
        super(DQN, self).__init__(
            training_envs, testing_envs, buffer, agent, logger,
            device, save_pth, args)

        algo_args = args.algorithm
        assert algo_args.name == "DQN", \
            "The method name in args must be 'DQN' for DQN algorithm."
        self.lr = algo_args.learning_rate
        self.gamma = algo_args.gamma
        self.start_epsilon = algo_args.start_epsilon
        self.epsilon = algo_args.start_epsilon
        self.end_epsilon = algo_args.end_epsilon
        self.epsilon_timestep = algo_args.epsilon_timestep
        self.epsilon_schedular = algo_args.epsilon_schedular
        self.use_target = algo_args.use_target
        self.batch_size = algo_args.batch_size
        self.device = device

        if self.use_target and target_agent is not None:
            self.target_agent = target_agent.to(self.device)
            self._target_hard_update()

        self.target_update_method = algo_args.target_update_method
        self.target_update_interval = algo_args.target_update_interval
        self.tau = algo_args.target_update_tau

        self.optimizer: torch.optim.Optimizer = \
            OPTIMIZER_DICT[algo_args.optimizer](
                self.agent.parameters(), lr=self.lr)

    # ------------------------------------------------------------------
    # Target network updates
    # ------------------------------------------------------------------

    def _target_hard_update(self):
        """Copy the online network's weights into the target network."""
        self.target_agent.load_state_dict(self.agent.state_dict())

    def _target_soft_update(self):
        """Polyak averaging:
        θ⁻ ← τ·θ + (1 − τ)·θ⁻
        """
        with torch.no_grad():
            for target_param, online_param in zip(
                    self.target_agent.parameters(),
                    self.agent.parameters()):
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(self.tau * online_param.data)

    # ------------------------------------------------------------------
    # Buffer / interaction
    # ------------------------------------------------------------------

    def _update_buffer(self, batch):
        self.buffer.add(batch)

    def interact_with_envs(self):
        batch, result = super().interact_with_envs()
        result.add_metric("epsilon", self.epsilon)
        return batch, result

    # ------------------------------------------------------------------
    # Policy update (the DQN loss)
    # ------------------------------------------------------------------

    def _update_policy(self):
        """
        Compute the DQN loss and take one gradient step.
        Implements the target from van Hasselt et al. (2016), Eq. 3–5:
            DQN:        Y_t = R_{t+1} + gamma · max_a Q(S_{t+1}, a; θ⁻)
            Double DQN: Y_t = R_{t+1} + gamma · Q(S_{t+1},
                                             argmax_a Q(S_{t+1}, a; θ); θ⁻)
        """
        # ------------------------------------------------------------------
        # 1. Sample a minibatch from the replay buffer
        # ------------------------------------------------------------------
        # Assumes buffer.sample(batch_size) returns a tuple of tensors:
        #   states, actions, rewards, next_states, dones
        # with shapes:
        #   states:      (B, 4, 84, 84)
        #   actions:     (B,) or (B, 1)
        #   rewards:     (B,) or (B, 1)
        #   next_states: (B, 4, 84, 84)
        #   dones:       (B,) or (B, 1) — 1 if terminal, 0 otherwise
        states, actions, rewards, next_states, dones = \
            self.buffer.sample(self.batch_size)


        # Convert to tensors first, THEN move to device
        states = torch.from_numpy(states).to(self.device, dtype=torch.float32)
        next_states = torch.from_numpy(next_states).to(self.device, dtype=torch.float32)
        actions = torch.from_numpy(actions).to(self.device, dtype=torch.long)
        rewards = torch.from_numpy(rewards).to(self.device, dtype=torch.float32)
        dones = torch.from_numpy(dones).to(self.device, dtype=torch.float32)

        # ------------------------------------------------------------------
        # 2. Compute Q(s, a) for the current states and actions
        # ------------------------------------------------------------------
        q = self.agent.get_q(states, actions)   # (B, 1)

        # ------------------------------------------------------------------
        # 3. Compute the TD target
        # ------------------------------------------------------------------
        with torch.no_grad():
            if self.use_target:
                # ---- Double DQN target ----
                # Selection: online network  θ
                # Evaluation: target network θ⁻
                next_q_online = self.agent.network(next_states)       # (B, A)
                best_actions = next_q_online.argmax(dim=1, keepdim=True)
                next_q_target = self.target_agent.network(
                    next_states)  # (B, A)
                next_q = next_q_target.gather(1, best_actions)         # (B, 1)
            else:
                # ---- Standard DQN target ----
                next_q = self.agent.get_max_q(next_states)             # (B, 1)

            # Zero out bootstrap at terminal states
            target = rewards + self.gamma * next_q * (1.0 - dones)     # (B, 1)

        # ------------------------------------------------------------------
        # 4. Compute the loss
        # ------------------------------------------------------------------
        # The paper uses the Huber loss (smooth L1) for robustness.
        td_error = target - q                                          # (B, 1)
        loss = F.smooth_l1_loss(q, target)

        # ------------------------------------------------------------------
        # 5. Gradient step
        # ------------------------------------------------------------------
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping is standard for DQN and helps stability
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=10.0)
        self.optimizer.step()

        # ------------------------------------------------------------------
        # 6. Periodically update the target network
        # ------------------------------------------------------------------
        if self.use_target:
            if self.target_update_method == "hard":
                if self.gradient_step % self.target_update_interval == 0:
                    self._target_hard_update()
            elif self.target_update_method == "soft":
                self._target_soft_update()

        # ------------------------------------------------------------------
        # 7. Build result object for logging
        # ------------------------------------------------------------------
        result = Result("")
        result.add_metric("network/loss", loss.item())
        result.add_metric("td_error", td_error.mean().item())
        result.add_metric("q_value_mean", q.mean().item())
        result.add_metric("target_mean", target.mean().item())

        self.gradient_step += 1
        return result

    # ------------------------------------------------------------------
    # Exploration
    # ------------------------------------------------------------------

    def random_choose_action(self):
        """
        Epsilon-greedy exploration helper.

        Returns True if we should take a random action this step.
        The caller is expected to update self.epsilon according to
        self.epsilon_schedular.
        """
        return np.random.random() < self.epsilon
