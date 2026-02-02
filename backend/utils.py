from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import io

from models import model, device, class_names

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_image(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)

    predicted_label = class_names[predicted_class.item()]
    confidence_score = confidence.item() * 100

    return predicted_label, confidence_score
