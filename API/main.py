#------------------ API to Use .pth model -------------------------------------

# import os
# import torch
# import uvicorn
# import torch.nn as nn
# import torch.nn.functional as F
# from fastapi import FastAPI, File, UploadFile, HTTPException
# from torchvision import transforms
# from PIL import Image
# from io import BytesIO
# import clip
# import numpy as np

# # FastAPI app
# app = FastAPI()

# # Constants
# IMAGE_SIZE = 224
# CLASS_NAMES = ['dry', 'normal', 'oily']
# MODEL_PATH = r"C:\Users\shazi\OneDrive\Desktop\VS Code\fyp\Sample testing\Test 7\Batch8_LR0.001_OptimRAdam.pth"

# # Device configuration
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # Load CLIP model
# clip_model, _ = clip.load("ViT-B/32", device=device)

# # Define classifier model
# class CLIPSkinClassifier(nn.Module):
#     def __init__(self, clip_model, num_classes=3):
#         super(CLIPSkinClassifier, self).__init__()
#         self.encoder = clip_model.visual
#         self.classifier = nn.Sequential(
#             nn.Linear(self.encoder.output_dim, 256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, num_classes)
#         )

#     def forward(self, x):
#         features = self.encoder(x)
#         logits = self.classifier(features)
#         return logits

# # Load your trained classifier
# model = CLIPSkinClassifier(clip_model, num_classes=len(CLASS_NAMES)).to(device)
# model = model.float()  # ensure model uses float32
# model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# model.eval()

# # Preprocessing
# transform = transforms.Compose([
#     transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
#                          (0.26862954, 0.26130258, 0.27577711))
# ])

# # Helper to read and preprocess image
# def read_image(data) -> torch.Tensor:
#     try:
#         image = Image.open(BytesIO(data)).convert("RGB")
#         tensor = transform(image).unsqueeze(0).to(device)
#         tensor = tensor.float()  # Match model's float precision
#         return tensor
#     except Exception:
#         raise HTTPException(status_code=400, detail="Invalid image format")

# # Prediction endpoint
# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     image_bytes = await file.read()
#     image_tensor = read_image(image_bytes)

#     with torch.no_grad():
#         outputs = model(image_tensor)
#         probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]
#         predicted_index = np.argmax(probabilities)
#         predicted_class = CLASS_NAMES[predicted_index]
#         confidence = float(probabilities[predicted_index])

#     return {
#         "predicted_class": predicted_class,
#         "confidence_score": confidence,
#         "probabilities": {
#             CLASS_NAMES[i]: float(prob) for i, prob in enumerate(probabilities)
#         }
#     }

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8080)

#------------------------ API to use .keras model------------------------

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from keras.models import load_model
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Constants matching training
IMAGE_SIZE = 128
CLASS_NAMES = ['dry', 'normal','oily']

# Load model
try:
    MODEL = load_model(r"C:\Users\shazi\OneDrive\Desktop\VS Code\AI_Project\saved_models\best_model.keras")
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise RuntimeError("Model loading failed")


def read_file_as_image(data) -> np.ndarray:
    """
    Read and convert image data to numpy array.
    """
    try:
        image = Image.open(BytesIO(data))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return np.array(image)
    except Exception as e:
        logger.error(f"Error reading image: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image format")


def preprocess_image(image: np.ndarray) -> tf.Tensor:
    """
    Preprocess the image: convert to tensor, resize, normalize.
    """
    try:
        image = tf.convert_to_tensor(image, dtype=tf.float32)  # Convert to float32
        image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))  # Resize
        image = image / 255.0  # Normalize to [0, 1]
        return tf.expand_dims(image, axis=0)  # Add batch dimension
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise HTTPException(status_code=500, detail="Image preprocessing failed")


@app.post("/predict")
async def predict(file: UploadFile = File(None)):
    if file is None:
        logger.warning("No file provided in the request.")
        raise HTTPException(status_code=400, detail="No file provided")

    try:
        # Step 1: Read the uploaded file
        image_data = await file.read()
        image = read_file_as_image(image_data)
        logger.debug(f"Original image shape: {image.shape}")

        # Step 2: Preprocess the image
        image_tensor = preprocess_image(image)
        logger.debug(f"Image tensor shape: {image_tensor.shape}")

        # Step 3: Predict
        pred_array = MODEL.predict(image_tensor)
        logger.debug(f"Raw predictions: {pred_array}")

        # Step 4: Interpret results
        pred_class_index = np.argmax(pred_array[0])
        predicted_class = CLASS_NAMES[pred_class_index]
        confidence = float(np.max(pred_array[0]))

        return {
            "class": predicted_class,
            "confidence": confidence,
            "probabilities": {
                name: float(prob)
                for name, prob in zip(CLASS_NAMES, pred_array[0])
            }
        }
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction failed")


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080)
