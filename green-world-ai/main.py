import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import json
import uvicorn

# Constants
IMG_SIZE = 256
MODEL_FILENAME = "model_epoch_20.h5"
CLASS_INDICES_FILE = "class_indices.json"
TEMP_DIR = "temp_uploads"

# Ensure temporary uploads directory exists
os.makedirs(TEMP_DIR, exist_ok=True)

# Load model
model_path = os.path.join(os.getcwd(), MODEL_FILENAME)
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
model = load_model(model_path)

# Load class index mapping
if not os.path.isfile(CLASS_INDICES_FILE):
    raise FileNotFoundError(f"Class indices file not found at {CLASS_INDICES_FILE}")
with open(CLASS_INDICES_FILE, 'r') as f:
    class_indices = json.load(f)  # expects {"0": "Apple___Apple_scab", "1": "Tomato___Late_blight", ...}

# Convert keys to int for lookup
class_indices = {int(k): v for k, v in class_indices.items()}

# FastAPI app initialization
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper: preprocess image

def load_and_preprocess(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

# Prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Save upload
        temp_path = os.path.join(TEMP_DIR, file.filename)
        with open(temp_path, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Preprocess and predict
        img_array = load_and_preprocess(temp_path)
        preds = model.predict(img_array)[0]
        class_id = int(np.argmax(preds))
        raw_label = class_indices.get(class_id, "Unknown___Unknown")

        # Parse label into name and condition
        parts = raw_label.split("___")
        name = parts[0].replace('_', ' ')
        condition = parts[1].replace('_', ' ') if len(parts) > 1 else ''

        return {"name": name, "condition": condition}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error processing image: {str(e)}"})

# Run with Uvicorn
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
