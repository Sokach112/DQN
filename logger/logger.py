"""Console metrics with optional TensorBoard and Weights & Biases output."""


import os
from typing import Any

from utils.result import Result
from rich.console import Console
from rich.table import Table



def _print(training: bool, epoch: int, interaction_step: int, gradient_step: int, metrics: dict[str, Any]):
    """Render metrics and counters without mutating the input dictionary."""
    console = Console()
    if training:
        title = "[bold yellow]Training Info[/bold yellow]"
    else:
        title = "[bold green]Testing Info[/bold green]"
    table = Table(title=title)
    table.add_column("Metric", justify="left")
    table.add_column("Value", justify="right")

    metrics = dict(metrics)
    metrics['epoch'] = epoch
    metrics['interaction_step'] = interaction_step
    metrics['gradient_step'] = gradient_step

    for k in sorted(sorted(metrics.keys())):
        table.add_row(k, str(metrics[k]))

    console.print(table)



class Logger:
    def __init__(self, project_name:str, run_name:str, log_dir="logs", use_wandb=True, use_tensorboard=True):
        """Initialize the component and retain its configuration."""
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        if self.use_wandb:
            import wandb
            self.wandb = wandb
            wandb.init(project=project_name, name=run_name,dir=log_dir)

        if self.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=log_dir)

    def _log(self, key, value, step):
        """Write one scalar to enabled backends using the transition counter."""
        if self.use_wandb:
            self.wandb.log({key: value}, step=step)
        if self.use_tensorboard:
            self.writer.add_scalar(key, value, global_step=step)

    def log_train(self, epoch, interaction_step, gradient_step, train_result: Result):
        """Report training metrics and counters to console and enabled backends."""
        for k,v in train_result.metrics.items():
            if "network" in k:
                self._log(k, v, interaction_step)
            else:
                self._log(f"train/{k}", v, interaction_step) # for non-loss metrics, we log them with interaction steps, for loss metrics, we log them with gradient steps


        _print(training=True, epoch=epoch, interaction_step=interaction_step, gradient_step=gradient_step, metrics=train_result.metrics)

    def log_test(self, epoch, interaction_step, gradient_step, test_result: Result):
        """Report evaluation metrics at the current transition count."""
        for k,v in test_result.metrics.items():
            self._log(f"test/{k}", v, interaction_step)
        _print(training=False, epoch=epoch, interaction_step=interaction_step, gradient_step=gradient_step, metrics=test_result.metrics)


    def close(self):
        """Flush and close enabled logging backends."""
        if self.use_wandb:
            self.wandb.finish()
        if self.use_tensorboard:
            self.writer.close()
