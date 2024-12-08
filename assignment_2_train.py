import torch
import torch.nn as nn
import torch.nn.functional as funct
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

#Hyperparameters
BATCH_SIZE = 32
LR = 0.001
NUM_EPOCHS = 20

# CNN model based on LeNet
class Shoe_CNN(nn.Module):
    def __init__(self):
        super(Shoe_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 5, stride = 1, padding = 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size = 3, padding = 1)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.flattener = nn.Flatten()
        self.fc1 = nn.LazyLinear(128)
        self.fc2 = nn.Linear(128, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.pool(funct.relu(self.conv1(x)))
        x = self.pool(funct.relu(self.conv2(x)))
        x = self.pool(funct.relu(self.conv3(x)))
        x = self.flattener(x)
        x = funct.relu(self.fc1(x))
        x = funct.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Class made to apply transform to an existing dataset
class ApplyTransform(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Creates the training and validation dataloaders
def dataload_train(trainData_path):
    full_dataset = datasets.ImageFolder(root = trainData_path)

    # 80% Train dataset to 20% validation dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Augmentation of only training dataset
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p = 0.25),
        transforms.RandomVerticalFlip(p = 0.25),
        transforms.RandomRotation(degrees = 45),
        transforms.ColorJitter(brightness = 0.5, contrast = 0.5, saturation = 0.5, hue = 0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    
    # Validation dataset doesn't need augmentation
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    train_dataset = ApplyTransform(train_dataset, transform = train_transform)
    val_dataset = ApplyTransform(val_dataset, transform = val_transform)

    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE)

    return train_loader, val_loader

# Calculates both the accuracy and the binary cross entropy loss in one function
def calc_acc_BCEloss(data_loader, model, loss_f):
    model.eval()
    correct = 0
    total_obj = 0
    total_loss = 0

    with torch.no_grad():
        for images, labels in data_loader:
            labels = labels.float().unsqueeze(1)
            outputs = model(images)

            # Accuracy component of calculations
            predictions = torch.sigmoid(outputs) 
            predicted_labels = (predictions > 0.5).float()
            total_obj += labels.size(0)
            correct += (predicted_labels == labels).sum().item()

            # BCELoss component of calculations
            loss = loss_f(outputs, labels)
            total_loss += loss.item()

    accuracy = 100 * correct / total_obj
    BCELoss = total_loss / len(data_loader)

    return accuracy, BCELoss

# Function to plot either loss or accuracy
def plot_info(legend_1, legend_2, plot_type):
    plt.figure(figsize=(10, 5))
    plt.plot(legend_1, label=f'Training {plot_type}')
    plt.plot(legend_2, label=f'Validation {plot_type}')
    plt.title(f'Training vs Validation {plot_type}')
    plt.xlabel('Epochs')
    plt.ylabel(plot_type)
    if plot_type == "Accuracy":
        axis = plt.gca()
        axis.set_ylim(top = 100.01)
    plt.legend()
    plt.show()

def main():
    trainData_path = "data/train"
    train_loader, val_loader = dataload_train(trainData_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Shoe_CNN().to(device) # Set device
    loss_f = nn.BCEWithLogitsLoss() # Loss function, Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr = LR)

    # Initialized variables to track best model and validation Binary Cross Entropy Loss
    best_val_BCELoss = float('inf')
    best_model = None

    # For plotting accuracy and loss
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []

    # Training loop
    num_epochs = NUM_EPOCHS
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            labels = labels.float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_f(outputs, labels)
            loss.backward()
            optimizer.step()

        train_acc, train_BCELoss = calc_acc_BCEloss(train_loader, model, loss_f)
        train_accuracies.append(train_acc)
        train_losses.append(train_BCELoss)

        val_acc, val_BCELoss = calc_acc_BCEloss(val_loader, model, loss_f)
        val_accuracies.append(val_acc)
        val_losses.append(val_BCELoss)

        # Save best model based on validation BCELoss
        if val_BCELoss < best_val_BCELoss:
            best_model = model.state_dict()

        print(f"Epoch [{epoch+1}/{num_epochs}], " +
              f"Train Loss: {val_BCELoss:.4f}, Train Accuracy: {train_acc:.4f}%, " +
              f"Validation Loss: {train_BCELoss:.4f}, Validation Accuracy: {val_acc:.4f}%")
    
    model.load_state_dict(best_model) # Load the best model
    torch.save(model.state_dict(), r'shoe_classify.pth') # Save the best model as pth
    print("Best model saved as 'shoe_classify.pth'.")

    # Plot relevant information
    plot_info(train_accuracies, val_accuracies, "Accuracy")
    plot_info(train_losses, val_losses, "Loss")

if __name__ == "__main__":
    main()