from torchvision import datasets

from src.config import DATA_DIR
from src.utils import get_mnist_transform


def main() -> None:
    transform = get_mnist_transform()
    datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)


if __name__ == "__main__":
    main()
