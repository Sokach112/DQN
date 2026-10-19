"""Double DQN updates, target synchronization, and exploration scheduling."""

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
        """Initialize the component and retain its configuration."""
        super(DQN, self).__init__(
            training_envs, testing_envs, buffer, agent, logger,
            device, save_pth, args)

        algo_args = args.algorithm
        assert algo_args.name == "DQN", \
            "The method name in args must be 'DQN' for DQN algorithm."
        self.loss_name = algo_args.get("loss", "huber")
        if self.loss_name not in LOSS_DICT:
            raise ValueError("loss must be mse or huber")
        self.lr = algo_args.learning_rate
        self.gamma = algo_args.gamma
        self.start_epsilon = algo_args.start_epsilon
        self.epsilon = algo_args.start_epsilon
        self.end_epsilon = algo_args.end_epsilon
        self.epsilon_timestep = algo_args.epsilon_timestep
        self.epsilon_schedular = algo_args.get('epsilon_schedule', algo_args.get('epsilon_schedular', 'linear'))
        if self.epsilon_schedular not in ('linear', 'constant'):
            raise ValueError('epsilon_schedule must be linear or constant')
        if self.epsilon_timestep <= 0:
            raise ValueError('epsilon_timestep must be positive')
        self.use_target = algo_args.use_target
        self.double_dqn = algo_args.get("double_dqn", True)
        self.batch_size = algo_args.batch_size
        self.device = device
        # Optional Boltzmann exploration is opt-in; epsilon-greedy remains the
        # default assignment baseline. See AtariDQNAgent.configure_exploration.
        self.agent.configure_exploration(
            algo_args.get("action_selection", "epsilon_greedy"),
            algo_args.get("softmax_temperature", 1.0),
        )

        if self.use_target and target_agent is None:
            raise ValueError('use_target requires target_agent')
        if self.use_target and target_agent is not None:
            self.target_agent = target_agent.to(self.device)
            self._target_hard_update()

        self.target_update_method = algo_args.target_update_method
        self.target_update_interval = algo_args.target_update_interval
        self.tau = algo_args.target_update_tau
        if self.target_update_method not in ("hard", "soft") or self.target_update_interval <= 0 or not 0 <= self.tau <= 1:
            raise ValueError("Invalid target update configuration")

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
        """Append the collected time-major transitions to replay."""
        self.buffer.add(batch)

    def interact_with_envs(self):
        """Collect vector steps and report rollout metrics including epsilon."""
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
        # Replay returns a dictionary; scalar targets must be column vectors.
        batch = self.buffer.sample(self.batch_size)
        states = batch['states']
        next_states = batch['next_states']
        actions = torch.as_tensor(batch['actions'], device=self.device, dtype=torch.long)
        rewards = torch.as_tensor(batch['rewards'], device=self.device).reshape(-1, 1)
        dones = torch.as_tensor(batch['dones'], device=self.device).reshape(-1, 1)

        # ------------------------------------------------------------------
        # 2. Compute Q(s, a) for the current states and actions
        # ------------------------------------------------------------------
        q = self.agent.get_q(states, actions)   # (B, 1)

        # ------------------------------------------------------------------
        # 3. Compute the TD target
        # ------------------------------------------------------------------
        with torch.no_grad():
            if self.use_target:
                if self.double_dqn:
                    next_q_online = self.agent.q_values(next_states)
                    best_actions = next_q_online.argmax(dim=1, keepdim=True)
                    next_q = self.target_agent.q_values(next_states).gather(1, best_actions)
                else:
                    next_q = self.target_agent.get_max_q(next_states)
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
        loss = LOSS_DICT[self.loss_name](q, target)

        # ------------------------------------------------------------------
        # 5. Gradient step
        # ------------------------------------------------------------------
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping is standard for DQN and helps stability
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=10.0)
        collect_metrics = getattr(self, '_collect_update_metrics', True)
        if collect_metrics:
            encoder_grad_norm = torch.stack([
                layer.weight.grad.detach().norm()
                for layer in (self.agent.network.conv1, self.agent.network.conv2, self.agent.network.conv3)
                if hasattr(layer, "weight") and layer.weight.grad is not None
            ]) if self.agent.network.conv1.weight.grad is not None else torch.zeros(1, device=self.device)
        self.optimizer.step()

        # ------------------------------------------------------------------
        # 6. Periodically update the target network
        # ------------------------------------------------------------------
        self.gradient_step += 1
        if self.use_target:
            if self.target_update_method == "hard":
                if self.gradient_step % self.target_update_interval == 0:
                    self._target_hard_update()
            elif self.target_update_method == "soft":
                self._target_soft_update()

        # ------------------------------------------------------------------
        # 7. Build result object for logging
        # ------------------------------------------------------------------
        # Scalar extraction from MPS/CUDA tensors forces device synchronization.
        # Skip it on non-logging updates; training behavior is unaffected.
        result = Result("")
        if collect_metrics:
            result.add_metric("network/loss", loss.item())
            result.add_metric("td_error", td_error.mean().item())
            result.add_metric("td_error_abs", td_error.abs().mean().item())
            result.add_metric("network/encoder_gradient_norm", encoder_grad_norm.norm().item())
            result.add_metric("q_value_mean", q.mean().item())
            result.add_metric("target_mean", target.mean().item())

        return result

    # ------------------------------------------------------------------
    # Exploration
    # ------------------------------------------------------------------

    def random_choose_action(self):
        """
        Epsilon-greedy exploration helper.

        Returns True if we should take a random action this step.
        The rollout updates epsilon after each vector step.
        """
        return np.random.random() < self.epsilon

    def update_epsilon(self):
        """Anneal exploration by total collected transitions, clamped at the end."""
        fraction = min(1.0, self.interaction_step / self.epsilon_timestep)
        self.epsilon = (self.start_epsilon + fraction * (self.end_epsilon - self.start_epsilon)
                        if self.epsilon_schedular == 'linear' else self.start_epsilon)
