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
        temp_filename = os.path.join(tempfile.gettempdir(), f"temp_{uuid.uuid4().hex}_{image.filename}")
        async with aiofiles.open(temp_filename, 'wb') as out_file:
            content = await image.read()
            await out_file.write(content)

        # Predict using Gradio client
        result = await client.predict(  # remove await if client.predict is not async
            image=handle_file(temp_filename),
            api_name="/predict"
        )
        # After prediction
        label = result['label'] if isinstance(result, dict) and 'label' in result else str(result)
        
        os.remove(temp_filename)

        return JSONResponse(content={"prediction": label})
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
