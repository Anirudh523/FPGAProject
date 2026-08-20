import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split

from conditiondataset import RoadConditionDataset
from model import RoadConditionCNN

file_path = "/Users/anirudh/RoadConditionClassifier/FPGAProject/ml/preprocessed/manifest.csv"
NUM_CLASSES = 4
IMAGE_SIZE = 128
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.2
RANDOM_SEED = 42
save_path = "/Users/anirudh/RoadConditionClassifier/FPGAProject/ml/best_model.pth"


def calculate_accuracy(logits, labels):
    predictions = logits.argmax(dim=1)
    correct = (predictions == labels).float().sum()
    total = labels.numel()
    return (correct / total).item()


def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    n_batches = 0
    total_accuracy = 0.0

    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_accuracy += calculate_accuracy(outputs, labels)
        n_batches += 1

    return running_loss / n_batches, total_accuracy / n_batches


def validate_one_epoch(model, dataloader, criterion):
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    n_batches = 0

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_accuracy += calculate_accuracy(outputs, labels)
            n_batches += 1

    return val_loss / n_batches, val_accuracy / n_batches


def main():
    full_dataset_no_aug = RoadConditionDataset(file_path, image_size=IMAGE_SIZE, augment=False)
    full_dataset_aug = RoadConditionDataset(file_path, image_size=IMAGE_SIZE, augment=True)

    dataset_size = len(full_dataset_no_aug)
    n_val = int(dataset_size * VAL_SPLIT)
    n_train = dataset_size - n_val

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_subset, val_subset = random_split(full_dataset_no_aug, [n_train, n_val], generator=generator)

    train_dataset = Subset(full_dataset_aug, train_subset.indices)
    val_dataset = val_subset

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = RoadConditionCNN(num_classes=NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_accuracy = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion)

        print(f"Epoch {epoch:2d}/{NUM_EPOCHS} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best val accuracy ({val_acc:.4f}), saved checkpoint to {save_path}")

    print(f"\nTraining complete. Best val accuracy: {best_val_accuracy:.4f}")


if __name__ == "__main__":
    main()