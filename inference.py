import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import pandas as pd
import argparse

from train import Shoe_CNN

# Class for creating the test dataset
class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')  # Convert to RGB
        if self.transform:
            image = self.transform(image)
        return image, self.images[idx]  # Return image and its filename

# Creates the test dataloader
def dataload_test(testData_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    
    test_dataset = TestDataset(root_dir = testData_path, transform = transform)
    test_loader = DataLoader(test_dataset, batch_size = 64)
    
    return test_loader

# Prints accuracy of model if given labels for test data
def printAccuracy(adidas_path, nike_path, predictions_df):
    correct = 0
    total = 0
    if adidas_path[-1] != '\\':
        adidas_path += '\\'
    if nike_path[-1] != '\\':
        nike_path += '\\'
    for idx in range(len(predictions_df['Filename'])):
        if predictions_df['Prediction'][idx] == "Adidas" and os.path.exists(f"{adidas_path}{predictions_df['Filename'][idx]}"):
            correct += 1
        elif predictions_df['Prediction'][idx] == "Nike" and os.path.exists(f"{nike_path}{predictions_df['Filename'][idx]}"):
            correct += 1
        total += 1
    print(f"Test Accuracy: {(100 * (correct / total)):.4f}")

def main(testData_path, adidas_path, nike_path):
    test_loader = dataload_test(testData_path)

    # load trained model
    model_path = "shoe_classify.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Shoe_CNN()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)

    # Run predictions of test data using model
    model.eval()
    predictions = []
    with torch.no_grad():
        for images, filenames in test_loader:
            outputs = model(images)
            predictions_batch = torch.sigmoid(outputs) # Restrict predictions within [0, 1]
            predicted_labels = (predictions_batch > 0.5).float()
            predictions.extend(zip(filenames, predicted_labels.numpy()))
    
    # Save the predictions to a CSV file
    predictions_df = pd.DataFrame(predictions, columns=['Filename', 'Prediction'])
    predictions_df['Prediction'] = predictions_df['Prediction'].apply(lambda x: 'Nike' if x[0] == 1 else 'Adidas')
    predictions_df.to_csv('predictions.csv', index=False)
    print("Predictions saved to 'predictions.csv'")

    # Prints accuracy of test data predictions only if labels are given through --labels_adidas and --labels_nike
    if adidas_path != None and nike_path != None:
        printAccuracy(adidas_path, nike_path, predictions_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference for shoe classification model')
    parser.add_argument('--data_path', type=str, help='path to test data')
    parser.add_argument('--labels_adidas', type=str, help='path to test data labels for adidas, if available')
    parser.add_argument('--labels_nike', type=str, help='path to test data labels for nike, if available')
    args = parser.parse_args()
    testData_path = args.data_path
    adidas_path = args.labels_adidas
    nike_path = args.labels_nike

    main(testData_path, adidas_path, nike_path)