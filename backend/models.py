import torch
import torch.nn as nn
from torchvision import models
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pth")

class_names = ["Cat", "Dog"]

def load_model(model_path, device):
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 2)
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model(MODEL_PATH, device)
