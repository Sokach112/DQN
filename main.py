"""Training entry point: configuration, seeding, resources, and algorithm assembly."""

import os
from pathlib import Path
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] ---------------%(message)s---------------',
    datefmt='%Y-%m-%d %H:%M:%S'
)

import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime

from env.make_envs import make_vec_envs
from algorithm.offpolicy_algorithm.baseoffpolicy import OffPolicyAlgorithm
from algorithm.offpolicy_algorithm import DQN
from algorithm.offpolicy_algorithm import ALGORITHM_DICT
from utils.utils import get_best_device, set_seed
from logger.logger import Logger
from memory import BUFFER_DICT
from agent.atari_agent import AtariDQNAgent


def get_args(cfg: DictConfig):

    """Record the working directory and display the resolved run configuration."""
    cfg.hydra_base_dir = os.getcwd()
    print("Parameters:", OmegaConf.to_yaml(cfg, resolve=True))
    return cfg

def resolve_checkpoint_path(args: DictConfig) -> Path:
    """Resolve optional save_pth as a file or directory, otherwise use log_dir.

    Existing directories, trailing separators, and suffixless paths are treated
    as directories and receive newest_model.pth. No checkpoint is loaded here.
    """
    configured = args.get("save_pth")
    if not configured:
        return (Path(args.log_dir).expanduser() / "newest_model.pth").resolve()
    path = Path(str(configured)).expanduser()
    if path.is_dir() or str(configured).endswith(("/", "\\")) or not path.suffix:
        path = path / "newest_model.pth"
    return path.resolve()


@hydra.main(config_path="config", config_name="config",version_base="1.3")
def main(cfg: DictConfig):
    """Build the training components and release environments and loggers on exit."""
    args = get_args(cfg)
    checkpoint_path = resolve_checkpoint_path(args)
    logging.info("Checkpoint destination: %s", checkpoint_path)
    set_seed(args.seed)

    training_envs = make_vec_envs(args.env,True,scale=False,seed=args.seed)

    logger = None
    try:
        logging.info("Checking for available GPUs...")
        device = get_best_device()

        logging.info("Creating the Logger...")
        runtime = datetime.now()
        runtime = runtime.strftime("%Y-%m-%d %H:%M:%S")
        logger = Logger(project_name=args.experiment_name, run_name=runtime, log_dir=args.log_dir, use_wandb=args.use_wandb, use_tensorboard=args.use_tensorboard)

        logging.info("Creating the ReplayBuffer...")
        buffer = BUFFER_DICT[args.algorithm.buffer_name](training_envs.observation_space, training_envs.action_space, args.algorithm.buffer_size, training_envs.num_envs)

        logging.info("Creating the Agent...")
        agent = AtariDQNAgent(training_envs.observation_space, training_envs.action_space, device, activation=args.algorithm.get("activation", "relu"), architecture=args.algorithm.get("architecture", "nature2015"))
        target_agent = AtariDQNAgent(training_envs.observation_space, training_envs.action_space, device, activation=args.algorithm.get("activation", "relu"), architecture=args.algorithm.get("architecture", "nature2015")) if args.algorithm.use_target else None


        # create trainer
        algorithm: OffPolicyAlgorithm | DQN

        algorithm = ALGORITHM_DICT[args.algorithm.name](training_envs=training_envs, testing_envs=None,
                                                   buffer=buffer, agent=agent,
                                                   logger=logger, device=device,
                                                   save_pth=str(checkpoint_path),
                                                   args=args,target_agent=target_agent)

        logging.info("Begin Training...")
        algorithm.run()

    finally:
        training_envs.close()
        if logger is not None:
            logger.close()


if __name__ == "__main__":
    main()
