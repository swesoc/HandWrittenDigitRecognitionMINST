import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

# download & load the dataset
transform = transforms.ToTensor()
#transform = transforms.Compose([
#    transforms.Grayscale(num_output_channels=3),
#    transforms.ToTensor()
#])


train_dataset = MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

val_dataset = MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Define Pytorcg CNN model
import torch.nn as nn
import torch.nn.functional as F


class MNIST_CNN(nn.Module):
    def __init__(
        self,
        use_dropout=False,
        extra_fc=False,
        extra_conv=False,
        dropout_p=0.5
    ):
        super().__init__()

        # ----- Convolutional layers -----
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        self.extra_conv = extra_conv
        if extra_conv:
            self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # ----- Fully connected layers -----
        conv_output_channels = 64 if extra_conv else 32
        self.fc1 = nn.Linear(conv_output_channels * 7 * 7, 128)

        self.extra_fc = extra_fc
        if extra_fc:
            self.fc_extra = nn.Linear(128, 64)

        self.fc2 = nn.Linear(64 if extra_fc else 128, 10)

        # ----- Dropout -----
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        if self.extra_conv:
            x = F.relu(self.conv3(x))

        x = x.reshape(x.size(0), -1)

        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = self.dropout(x)

        if self.extra_fc:
            x = F.relu(self.fc_extra(x))
            if self.use_dropout:
                x = self.dropout(x)

        x = self.fc2(x)
        return x

def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)

def evaluate_model(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
#create output directory

from pathlib import Path

model_dir = Path("models")
model_dir.mkdir(exist_ok=True)



experiments = [
    {"name": "Baseline", "use_dropout": False, "extra_fc": False, "extra_conv": False},
    {"name": "Dropout only", "use_dropout": True, "extra_fc": False, "extra_conv": False},
    {"name": "Extra Conv", "use_dropout": False, "extra_fc": False, "extra_conv": True},
    {"name": "All features", "use_dropout": True, "extra_fc": True, "extra_conv": True},
]

epochs = 5
results = {}
for exp in experiments:

    model = MNIST_CNN(
        use_dropout=exp["use_dropout"],
        extra_fc=exp["extra_fc"],
        extra_conv=exp["extra_conv"]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        loss = train_model(model, train_loader, optimizer, criterion, device)


    results[exp['name']] = evaluate_model(model, val_loader, device)


    dummy_input = torch.randn(1, 1, 28, 28)
    #export and saves the model
    torch.onnx.export(
        model,
        dummy_input,
        model_dir / f"{exp['name']}.onnx"
    )


    print(f"Saved: model_dir / {exp['name']}.onnx")


