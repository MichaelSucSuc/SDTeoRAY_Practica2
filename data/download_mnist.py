from torchvision import datasets, transforms

from src.config import DATA_DIR


def main() -> None:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)


if __name__ == "__main__":
    main()
