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
from torchvision import transforms, models

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Define class names in the correct order
class_names = ['Bellpepper_fresh', 'Bellpepper_intermediate_fresh', 'Bellpepper_rotten', 'Carrot_fresh', 'Carrot_intermediate_fresh', 'Carrot_rotten', 'Cucumber_fresh', 'Cucumber_intermediate_fresh', 'Cucumber_rotten', 'Potato_fresh', 'Potato_intermediate_fresh', 'Potato_rotten', 'Tomato_fresh', 'Tomato_intermediate_fresh', 'Tomato_rotten', 'ripe_apple', 'ripe_banana', 'ripe_mango', 'ripe_oranges', 'ripe_strawberry', 'rotten_apple', 'rotten_banana', 'rotten_mango', 'rotten_oranges', 'rotten_strawberry', 'unripe_apple', 'unripe_banana', 'unripe_mango', 'unripe_oranges', 'unripe_strawberry']
num_classes = len(class_names)

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


# Define the model architecture
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("best_model.pth", map_location=torch.device('cpu')))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


@app.post("/predict")
async def predict_image(image: UploadFile = File(...)):
    try:
        temp_filename = f"temp_{uuid.uuid4().hex}_{image.filename}"
        async with aiofiles.open(temp_filename, 'wb') as out_file:
            content = await image.read()
            await out_file.write(content)

        print(f"[INFO] Saved image as {temp_filename}")

        image = Image.open(temp_filename).convert('RGB')

        img_tensor = transform(image).unsqueeze(0)
        img_tensor = img_tensor.to(next(model.parameters()).device)

        with torch.no_grad():
            outputs = model(img_tensor)
            _, preds = torch.max(outputs, 1)
            predicted_class = class_names[preds.item()]

        print(f"[INFO] Gradio predicted_class: {predicted_class}")

        os.remove(temp_filename)

        return JSONResponse(content={"prediction": predicted_class})
    
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
