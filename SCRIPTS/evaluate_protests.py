import os
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
from torchvision import models

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModifiedResNet50(nn.Module):
    def __init__(self, num_classes=12):
        super(ModifiedResNet50, self).__init__()
        original_model = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.classifier = nn.Linear(original_model.fc.in_features, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return self.sigmoid(x)

class ImageDataset(Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(img_path)

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Set the directory where your images are stored
img_dir = r'C:\Users\drieb\Downloads\ucla-protest\UCLA-protest\img\train'
img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')]

# Load the dataset and create a DataLoader
dataset = ImageDataset(img_paths=img_paths, transform=transform)
data_loader = DataLoader(dataset, batch_size=16, shuffle=False)

# Load the model
model = ModifiedResNet50().to(device)
model.eval()  # Set the model to evaluation mode

# Evaluate the images
results = []
with torch.no_grad():
    for inputs, img_names in tqdm(data_loader, desc="Evaluating"):
        inputs = inputs.to(device)
        outputs = model(inputs)
        predictions = outputs.cpu().data.numpy()
        for img_name, prediction in zip(img_names, predictions):
            results.append((img_name, *prediction))

# Save evaluation results to a CSV file
df_results = pd.DataFrame(results, columns=['ImageName', 'Protest', 'Violence', 'Sign', 'Photo', 'Fire', 'Police', 'Children', 'Group_20', 'Group_100', 'Flag', 'Night', 'Shouting'])
output_csv_path = os.path.join(img_dir, 'evaluation_results.csv')
df_results.to_csv(output_csv_path, index=False)
print("Results saved to:", output_csv_path)
print(df_results.head())




