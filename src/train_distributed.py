from typing import Any

import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ray import train
from ray.air.config import CheckpointConfig, RunConfig, ScalingConfig
from ray.train import Checkpoint
from ray.train.torch import TorchConfig, TorchTrainer
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets

from src.config import BATCH_SIZE, DATA_DIR, EPOCHS, LEARNING_RATE, NUM_WORKERS, USE_GPU
from src.utils import get_mnist_transform


class SimpleMNISTModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def load_data(data_dir: str = DATA_DIR) -> tuple[datasets.MNIST, datasets.MNIST]:
    transform = get_mnist_transform()
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    return train_dataset, test_dataset


def train_func_per_worker(config: dict[str, Any]) -> None:
    epochs = config.get("epochs", EPOCHS)
    lr = config.get("lr", LEARNING_RATE)
    batch_size = config.get("batch_size", BATCH_SIZE)
    model = SimpleMNISTModel()
    device = torch.device("cuda" if torch.cuda.is_available() and USE_GPU else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_dataset, _ = load_data()
    context = train.get_context()
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=context.get_world_size(),
        rank=context.get_rank(),
        shuffle=True,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        train.report(
            metrics={"loss": avg_loss, "epoch": epoch},
            checkpoint=Checkpoint.from_dict({"model_state_dict": model.state_dict()}),
        )


def main() -> Any:
    try:
        ray.init(ignore_reinit_error=True, address="auto")
    except Exception:
        ray.init(ignore_reinit_error=True)

    scaling_config = ScalingConfig(
        num_workers=NUM_WORKERS,
        use_gpu=torch.cuda.is_available() and USE_GPU,
        trainer_resources={"CPU": 1},
    )
    torch_config = TorchConfig(backend="nccl" if torch.cuda.is_available() and USE_GPU else "gloo")
    train_config = {"epochs": EPOCHS, "lr": LEARNING_RATE, "batch_size": BATCH_SIZE}

    trainer = TorchTrainer(
        train_loop_per_worker=train_func_per_worker,
        train_loop_config=train_config,
        scaling_config=scaling_config,
        torch_config=torch_config,
        run_config=RunConfig(
            checkpoint_config=CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute="loss",
                checkpoint_score_order="min",
            )
        ),
    )
    result = trainer.fit()
    return result


if __name__ == "__main__":
    main()
