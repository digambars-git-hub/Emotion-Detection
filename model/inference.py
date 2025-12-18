# inference.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load checkpoint
checkpoint = torch.load("emotion_recognition_model.pth", map_location=device)
classes = checkpoint["classes"]

# recreate model
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(1280, len(classes))
model.load_state_dict(checkpoint["model_state"])

model.to(device)
model.eval()

# preprocessing (SAME as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@torch.no_grad()
def predict(pil_image: Image.Image):
    x = transform(pil_image).unsqueeze(0).to(device)
    logits = model(x)
    idx = logits.argmax(dim=1).item()
    return classes[idx]

if __name__ == "__main__":
    img = Image.open("Image.jpg")  # any face image
    print(predict(img))
