import torch
from torchvision import transforms, models
from PIL import Image

# Configs
model_path = r"C:\Users\shazi\OneDrive\Desktop\VS Code\fyp\resnet18 model\best_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['dry', 'normal', 'oily']  # order from ImageFolder

# Define transform (same as validation/test)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load model
model = models.resnet18(weights=None)  # weights=None to avoid re-downloading
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Predict function
def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        predicted_class = class_names[predicted.item()]

    print(f"✅ Predicted Skin Type: {predicted_class.capitalize()}")


# Example usage
image_path = r'fyp\resnet18 model\2.jpg'
predict_image(image_path)
