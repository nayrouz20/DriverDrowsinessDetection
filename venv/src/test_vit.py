import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('************GPU***********')

# Define the data transformations for testing
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define the data directory and load the test dataset
data_dir = 'C:/Users/linda/OneDrive/Desktop/images/test'  # Replace with the path to your dataset
test_dataset = datasets.ImageFolder(data_dir, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Load the pre-trained ViT model and the trained weights
model_path = 'C:/Users/linda/OneDrive/Desktop/trainvit/vit_finetuned.pth'  # Replace with the path to your saved model weights
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=2)
model.load_state_dict(torch.load(model_path, map_location=device))  # Load the model weights
model = model.to(device)

# Function to test the model
def test_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs).logits
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_labels, all_preds

# Run the test
if __name__ == '__main__':
    labels, preds = test_model(model, test_loader)

    # Calculate accuracy
    accuracy = accuracy_score(labels, preds)
    print(f'Test Accuracy: {accuracy:.4f}')

    # Print classification report
    class_names = test_dataset.classes
    print(classification_report(labels, preds, target_names=class_names))

    # Create confusion matrix
    cm = confusion_matrix(labels, preds)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
