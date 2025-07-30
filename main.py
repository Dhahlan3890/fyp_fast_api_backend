from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from gradio_client import Client, handle_file
import aiofiles
import os
import uuid
import tempfile
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from efficientnet_pytorch import EfficientNet

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# client = Client("Dhahlan2000/predict_freshness_and_ripeness")
client = Client("Dhahlan2000/level1_freshness_classifier")


class RipenessClassifier(nn.Module):
    def __init__(self, num_classes):
        super(RipenessClassifier, self).__init__()
        self.base_model = EfficientNet.from_pretrained('efficientnet-b3')
        num_features = self.base_model._fc.in_features
        self.base_model._fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

def load_model(model_path, num_classes, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = RipenessClassifier(num_classes).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def predict(image_path, model, class_names, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Preprocess
    input_tensor = preprocess_image(image_path).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
    
    # Print all class probabilities
    print("\nClass Probabilities:")
    for i, prob in enumerate(probabilities):
        print(f"{class_names[i]}: {prob:.4f}")
    
    return class_names[predicted_idx], confidence


@app.post("/predict")
async def predict_image(image: UploadFile = File(...)):
    try:
        temp_filename = f"temp_{uuid.uuid4().hex}_{image.filename}"
        async with aiofiles.open(temp_filename, 'wb') as out_file:
            content = await image.read()
            await out_file.write(content)

        print(f"[INFO] Saved image as {temp_filename}")

        # ❌ Don't use 'await' here — it's not an async function
        result = client.predict(
            image=handle_file(temp_filename),
            api_name="/predict"
        )

        print(f"[INFO] Gradio result: {result}")

        label = result['label'] if isinstance(result, dict) else str(result)

        if label == 'Bellpepper':
            MODEL_PATH = 'best_bellpepper_model.pth'
            CLASS_NAMES = ['fresh', 'intermediate', 'rotten']  # Replace with your actual class names
            NUM_CLASSES = len(CLASS_NAMES)
            model = load_model(MODEL_PATH, NUM_CLASSES)
            predicted_class, confidence = predict(temp_filename, model, CLASS_NAMES)

        elif label == 'Carrot':
            MODEL_PATH = 'best_carrot_model.pth'
            CLASS_NAMES = ['fresh', 'intermediate', 'rotten']
            NUM_CLASSES = len(CLASS_NAMES)
            model = load_model(MODEL_PATH, NUM_CLASSES)
            predicted_class, confidence = predict(temp_filename, model, CLASS_NAMES)

        elif label == 'Cucumber':
            MODEL_PATH = 'best_cucumber_model.pth'
            CLASS_NAMES = ['fresh', 'intermediate', 'rotten']
            NUM_CLASSES = len(CLASS_NAMES)
            model = load_model(MODEL_PATH, NUM_CLASSES)
            predicted_class, confidence = predict(temp_filename, model, CLASS_NAMES)

        elif label == 'Potato':
            MODEL_PATH = 'best_potato_model.pth'
            CLASS_NAMES = ['fresh', 'intermediate', 'rotten']
            NUM_CLASSES = len(CLASS_NAMES)
            model = load_model(MODEL_PATH, NUM_CLASSES)
            predicted_class, confidence = predict(temp_filename, model, CLASS_NAMES)

        elif label == 'Tomato':
            MODEL_PATH = 'best_tomato_model.pth'
            CLASS_NAMES = ['fresh', 'intermediate', 'rotten']
            NUM_CLASSES = len(CLASS_NAMES)
            model = load_model(MODEL_PATH, NUM_CLASSES)
            predicted_class, confidence = predict(temp_filename, model, CLASS_NAMES)

        elif label == 'Apple':
            MODEL_PATH = 'best_apple_model.pth'
            CLASS_NAMES = ['ripe', 'rotten', 'unripe']
            NUM_CLASSES = len(CLASS_NAMES)
            model = load_model(MODEL_PATH, NUM_CLASSES)
            predicted_class, confidence = predict(temp_filename, model, CLASS_NAMES)

        elif label == 'Banana':
            MODEL_PATH = 'best_banana_model.pth'
            CLASS_NAMES = ['ripe', 'rotten', 'unripe']
            NUM_CLASSES = len(CLASS_NAMES)
            model = load_model(MODEL_PATH, NUM_CLASSES)
            predicted_class, confidence = predict(temp_filename, model, CLASS_NAMES)

        elif label == 'mango':
            MODEL_PATH = 'best_mango_model.pth'
            CLASS_NAMES = ['ripe', 'rotten', 'unripe']
            NUM_CLASSES = len(CLASS_NAMES)
            model = load_model(MODEL_PATH, NUM_CLASSES)
            predicted_class, confidence = predict(temp_filename, model, CLASS_NAMES)

        elif label == 'oranges':
            MODEL_PATH = 'best_orange_model.pth'
            CLASS_NAMES = ['ripe', 'rotten', 'unripe']
            NUM_CLASSES = len(CLASS_NAMES)
            model = load_model(MODEL_PATH, NUM_CLASSES)
            predicted_class, confidence = predict(temp_filename, model, CLASS_NAMES)

        elif label == 'strawberry':
            MODEL_PATH = 'best_strawberry_model.pth'
            CLASS_NAMES = ['ripe', 'rotten', 'unripe']
            NUM_CLASSES = len(CLASS_NAMES)
            model = load_model(MODEL_PATH, NUM_CLASSES)
            predicted_class, confidence = predict(temp_filename, model, CLASS_NAMES)

        else:
            return JSONResponse(status_code=400, content={"error": "Unsupported label"})

        os.remove(temp_filename)
        return JSONResponse(content={"prediction": f"{predicted_class}_{label}"})

    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/natural-artificial")
async def predict_image(image: UploadFile = File(...)):
    try:
        temp_filename = f"temp_{uuid.uuid4().hex}_{image.filename}"
        async with aiofiles.open(temp_filename, 'wb') as out_file:
            content = await image.read()
            await out_file.write(content)

        print(f"[INFO] Saved image as {temp_filename}")

        # ❌ Don't use 'await' here — it's not an async function
        result = client.predict(
            image=handle_file(temp_filename),
            api_name="/predict"
        )

        print(f"[INFO] Gradio result: {result}")

        label = result['label'] if isinstance(result, dict) else str(result)


        if label == 'Banana':
            
            new_result = client1.predict(
                    image=handle_file(temp_filename),
                    api_name="/predict"
            )
            print(new_result)
            predicted_class = new_result['label'] if isinstance(new_result, dict) else str(new_result)

        elif label == 'mango':

            new_result = client2.predict(
                    image=handle_file(temp_filename),
                    api_name="/predict"
            )
            print(new_result)
            predicted_class = new_result['label'] if isinstance(new_result, dict) else str(new_result)

        else:
            return JSONResponse(status_code=400, content={"error": "Unsupported label"})

        os.remove(temp_filename)
        return JSONResponse(content={"prediction": f"{predicted_class}_{label}"})

    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

