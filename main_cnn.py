from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from gradio_client import Client, handle_file
import aiofiles
import os
import uuid
import tempfile
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configuration
IMG_SIZE = 128
MODEL_PATH = "ripeness_cnn_model_aug.pth"  # Path to the saved model

# Define class names in the correct order
class_names = [
    "Fresh Bellpepper",
    "Intermediate Fresh Bellpepper",
    "Rotten Bellpepper",
    "Fresh Carrot",
    "Intermediate Fresh Carrot",
    "Rotten Carrot",
    "Fresh Cucumber",
    "Intermediate Fresh Cucumber",
    "Rotten Cucumber",
    "Fresh Potato",
    "Intermediate Fresh Potato",
    "Rotten Potato",
    "Fresh Tomato",
    "Intermediate Fresh Tomato",
    "Rotten Tomato",
    "Ripe Apple",
    "Ripe Banana",
    "Ripe Mango",
    "Ripe Orange",
    "Ripe Strawberry",
    "Rotten Apple",
    "Rotten Banana",
    "Rotten Mango",
    "Rotten Orange",
    "Rotten Strawberry",
    "Unripe Apple",
    "Unripe Banana",
    "Unripe Mango",
    "Unripe Orange",
    "Unripe Strawberry"
]

# Define transforms
val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Define the model architecture
class RipenessCNN(nn.Module):
    def __init__(self, num_classes):
        super(RipenessCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 32 * 32, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 64x64
        x = self.pool(F.relu(self.conv2(x)))  # 32x32
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Load model
model = RipenessCNN(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


@app.post("/predict")
async def predict_image(image: UploadFile = File(...)):
    try:
        temp_filename = f"temp_{uuid.uuid4().hex}_{image.filename}"
        async with aiofiles.open(temp_filename, 'wb') as out_file:
            content = await image.read()
            await out_file.write(content)

        print(f"[INFO] Saved image as {temp_filename}")

        image = Image.open(temp_filename).convert('RGB')

        image_tensor = val_transforms(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            _, preds = torch.max(outputs, 1)
            predicted_class = class_names[preds.item()]

        print(f"[INFO] Gradio predicted_class: {predicted_class}")

        os.remove(temp_filename)

        return JSONResponse(content={"prediction": predicted_class})
    
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
