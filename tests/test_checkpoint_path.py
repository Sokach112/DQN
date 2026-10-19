"""Checkpoint destination overrides must reach the training algorithm."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from omegaconf import OmegaConf
from main import main, resolve_checkpoint_path


class CheckpointPathTests(unittest.TestCase):
    def test_default_directory_and_file(self):
        """Preserve default output while supporting Colab directory overrides."""
        with tempfile.TemporaryDirectory() as root:
            cfg = OmegaConf.create({'log_dir': root})
            self.assertEqual(resolve_checkpoint_path(cfg), Path(root).resolve()/'newest_model.pth')
            for destination in (root+'/drive/', root+'/drive', root):
                cfg.save_pth = destination
                self.assertEqual(resolve_checkpoint_path(cfg), Path(destination).resolve()/'newest_model.pth')
            cfg.save_pth = root+'/model.pth'
            self.assertEqual(resolve_checkpoint_path(cfg), Path(root).resolve()/'model.pth')

    def test_entrypoint_uses_override(self):
        """Check wiring through main, not just path resolution in isolation."""
        cfg = OmegaConf.load('config/config.yaml')
        cfg.log_dir = '/tmp/rl-checkpoint-test-logs'
        del cfg.hydra
        cfg.save_pth = '/tmp/rl-checkpoint-test/'
        factory = Mock()
        with patch('main.make_vec_envs'), patch('main.get_best_device',return_value='cpu'), patch('main.Logger'), patch('main.BUFFER_DICT', {'ReplayBuffer': Mock()}), patch('main.AtariDQNAgent'), patch('main.ALGORITHM_DICT', {'DQN': factory}):
            main.__wrapped__(cfg)
        self.assertEqual(factory.call_args.kwargs['save_pth'], str(Path('/tmp/rl-checkpoint-test/newest_model.pth').resolve()))
        factory.return_value.run.assert_called_once()
