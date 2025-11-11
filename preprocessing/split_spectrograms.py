from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch
import os

SCRIPT_PATH = os.getcwd()
INPUT_DIR = os.path.abspath(os.path.join(SCRIPT_PATH, '..', 'data', 'spectrograms'))
OUTPUT_DIR = os.path.join(SCRIPT_PATH, '..', 'data', 'spectrograms')

# transformacje wstępne: normalizacja, augmentacja (opcjonalna)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # zmniejszenie spektrogramu
    transforms.Grayscale(num_output_channels=1),  # 1 kanał (lub 3 dla RGB)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# dataset z katalogów
dataset = datasets.ImageFolder(root=INPUT_DIR, transform=transform)

# podział: np. 80% train, 20% test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(train_dataset)
print(f"Liczba obrazów treningowych: {len(train_dataset)}, testowych: {len(test_dataset)}")
