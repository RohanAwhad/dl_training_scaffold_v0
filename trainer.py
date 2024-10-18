import argparse
import yaml
import os
import torch
from torch.utils.data import DataLoader
import logging
import sys
import wandb
from abc import ABC, abstractmethod

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
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

# === Argument Parser for config file === #
def parse_args():
    parser = argparse.ArgumentParser(description="Trainer Scaffold")
    parser.add_argument('--config', type=str, required=True, help="Path to the config YAML file")
    args = parser.parse_args()
    return args

# === Placeholder functions for later use === #
def build_model(model_name, pretrained_flag):
    """Load or initialize a model (placeholder)."""
    raise NotImplementedError('build_model is not implemented')
    model = None  # Placeholder: Replace with model initialization logic
    return model

def load_dataset(dataset_path, shard_size):
    """Placeholder for dataset loading and sharding logic."""
    raise NotImplementedError('load_dataset is not implemented')
    train_dataset, val_dataset = None, None  # Placeholder
    return train_dataset, val_dataset

def evaluate(model, val_loader, step, logger):
    """Placeholder for evaluation logic."""
    model.eval()
    raise NotImplementedError('evaluate is not implemented')
    # Example: logging evaluation metrics using logger
    eval_metrics = {"accuracy": 0.9}  # Placeholder for evaluation metrics
    logger.log(eval_metrics, step)

def save_checkpoint(model, optimizer, ckpt_dir, step):
    """Placeholder for checkpointing logic."""
    model.eval()
    # TODO: also save current step in the dict
    raise NotImplementedError('save_checkpoint is not implemented')
    ckpt_path = os.path.join(ckpt_dir, f"checkpoint_step_{step}.pth")
    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, ckpt_path)
    print(f"Checkpoint saved at step {step}")

# === Trainer Scaffold === #
def main():
    # Parse config file argument
    args = parse_args()

    # Load YAML config
    config = load_config(args.config)

    # Set up the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Set up WandbLogger
    logger = WandbLogger(project_name=config.get('project_name', 'default_project'), run_name=config['run_name'])
    logger.log({"status": "Starting run"}, step=0)

    # Build model
    model = build_model(config['model_name'], config['pretrained_flag'])
    model = model.to(device)

    # Load dataset and DataLoader
    train_dataset, val_dataset = load_dataset(config['dataset_path'], config['shard_size'])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=True,  # Hardcoded based on your instructions
        prefetch_factor=config['prefetch_factor']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=True,  # Hardcoded based on your instructions
        prefetch_factor=config['prefetch_factor']
    )
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
