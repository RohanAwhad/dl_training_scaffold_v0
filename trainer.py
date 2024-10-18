import argparse
import functools
import numpy as np
import os
import sys
import wandb
import yaml

from abc import ABC, abstractmethod
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# === Logger Classes === #
class Logger(ABC):
    @abstractmethod
    def log(self, data: dict, step: int):
        pass

class WandbLogger(Logger):
    def __init__(self, project_name, run_name):
        self.run = wandb.init(project=project_name, name=run_name)

    def log(self, data: dict, step: int):
        self.run.log(data, step=step)

# === Utility to load YAML configuration === #
def load_config(config_file):
    if len(sys.argv) < 2:
        print('Please provide a config.yaml path as arg')
        exit(0)
    config_file = sys.argv[1]
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)


@functools.lru_cache(maxsize=2)
def load_shard(filename: str) -> np.ndarray: return np.load(filename)

class ShardedDataset(Dataset):
    def __init__(self, shard_paths: list[str], shard_size: int):
        super().__init__()
        self.shard_paths = shard_paths
        self.shard_size = shard_size
    def __len__(self): return len(self.shard_paths)*self.shard_size
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        shard_idx = index // self.shard_size
        shard = load_shard(self.shard_paths[shard_idx])
        example = shard[index % self.shard_size] 

        raise NotImplementedError('__getitem__ is not implemented in ShardedDataset')
        image, label = example[:3], example[3:]
        return image, label


def build_model(model_name: str, pretrained_flag: bool) -> nn.Module:
    """Load or initialize a model (placeholder)."""
    raise NotImplementedError('build_model is not implemented')
    model = None  # Placeholder: Replace with model initialization logic
    return model

def criterion(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor | float]:
    """Returns a dict with the final loss in key 'total_loss'"""
    raise NotImplementedError('criterion is not implemented')

def load_dataset(dataset_path: str, shard_size: int) -> Tuple[Dataset, Dataset]:
    """Placeholder for dataset loading and sharding logic."""
    raise NotImplementedError('load_dataset is not implemented')
    train_dataset, val_dataset = None, None  # Placeholder
    return train_dataset, val_dataset

def evaluate(model: nn.Module, val_loader: DataLoader, step: int, logger: Logger) -> None:
    """Placeholder for evaluation logic."""
    model.eval()
    raise NotImplementedError('evaluate is not implemented')
    eval_metrics = {"accuracy": 0.9}  # Placeholder for evaluation metrics
    logger.log(eval_metrics, step)

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, ckpt_dir: str, step: int) -> None:
    """Placeholder for checkpointing logic."""
    model.eval()
    raise NotImplementedError('save_checkpoint is not implemented')
    ckpt_path = os.path.join(ckpt_dir, f"checkpoint_step_{step}.pth")
    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, ckpt_path)
    print(f"Checkpoint saved at step {step}")

# === Trainer Scaffold === #
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    config = load_config()
    logger = WandbLogger(project_name=config.get('project_name', 'default_project'), run_name=config['run_name'])
    model = build_model(config['model_name'], config['pretrained_flag'])
    model = model.to(device)
    print('Model has been loaded on device')
    train_dataset, val_dataset = load_dataset(config['dataset_path'], config['shard_size'])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=True,
        prefetch_factor=config['prefetch_factor']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=True,  # Hardcoded based on your instructions
        prefetch_factor=config['prefetch_factor']
    )
    print('DataLoaders created')
    # TODO: calculate save, and update where you are in the dataloader so you can restart training

    # Optimizer placeholder
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])

    # Main training loop
    step = 0
    while step < config['num_steps']:
        for batch in train_loader:
            step += 1

            if step % config['eval_interval'] == 0:
                print(f"Step {step}: Performing evaluation")
                evaluate(model, val_loader, step, logger)

            if step % config['ckpt_interval'] == 0 and config['do_ckpt']:
                print(f"Step {step}: Saving checkpoint")
                save_checkpoint(model, optimizer, config['ckpt_dir'], step)
            
            # Placeholder for model training step
            model.train()
            raise NotImplementedError('training loop is not implemented')
            # Example: output = model(batch['input'])
            # loss = compute_loss(output, batch['target'])
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # Log training loss or metrics using the WandbLogger
            training_metrics = {"loss": 0.1}  # Placeholder for actual loss value
            logger.log(training_metrics, step)

            if step >= config['num_steps']:
                break

    logger.log({"status": "Training finished"}, step=step)

if __name__ == "__main__":
    main()
