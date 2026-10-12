"""Scalar metric aggregation and timed training/evaluation sections."""

import time
import numpy as np
import torch
import warnings
class Result:
    def __init__(self, head:str):
        """Initialize the component and retain its configuration."""
        self.metrics = {}
        self.start_time = None
        self.end_time = None
        self.head = head
    def __enter__(self):
        """Start the timer and return this metric collection."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Record elapsed wall time without suppressing exceptions."""
        self.end_time = time.perf_counter()
        self.metrics[f"{self.head}/time"] = self.end_time - self.start_time


    def add_metric(self, name, value):
        """Add a unique scalar metric, reducing nonempty arrays or tensors by mean."""
        if name in self.metrics:
            raise ValueError(f"Metric {name} already exists in the result, you are overwriting it.")

        if isinstance(value, (np.ndarray, np.generic)):
            if value.size == 0:
                raise ValueError('Cannot log an empty array')
            value = float(np.asarray(value).mean())
        elif isinstance(value, torch.Tensor):
            if value.numel() == 0:
                raise ValueError('Cannot log an empty tensor')
            value = value.detach().float().mean().item()
        elif isinstance(value, (int, float)):
            value = float(value)
        else:
            raise ValueError(f'Unsupported metric type: {type(value)}')
        self.metrics[name] = value

    def add(self, result):
        """group other result into the current result"""
        for k,v in result.metrics.items():
            self.add_metric(k,v)
