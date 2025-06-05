from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from gradio_client import Client, handle_file
import aiofiles
import os
import uuid
import tempfile

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Client("Dhahlan2000/predict_freshness_and_ripeness")

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

        os.remove(temp_filename)

        label = result['label'] if isinstance(result, dict) else str(result)
        return JSONResponse(content={"prediction": label})
    
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
