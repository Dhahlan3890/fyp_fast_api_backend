from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from gradio_client import Client, handle_file
import aiofiles
import os
import uuid

app = FastAPI()

# Allow all origins (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gradio client
client = Client("Dhahlan2000/predict_freshness_and_ripeness")

@app.post("/predict")
async def predict_image(image: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        temp_filename = f"temp_{uuid.uuid4().hex}_{image.filename}"
        async with aiofiles.open(temp_filename, 'wb') as out_file:
            content = await image.read()
            await out_file.write(content)

        # Use Gradio client to predict
        result = client.predict(
            image=handle_file(temp_filename),
            api_name="/predict"
        )

        # Remove temp file
        os.remove(temp_filename)

        # Return result
        return JSONResponse(content={"prediction": result})
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
