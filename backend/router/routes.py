from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse

from utils import predict_image

router = APIRouter()

@router.get("/")
def root():
    return {"status": "FastAPI backend running"}

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    predicted_label, confidence_score = predict_image(image_bytes)

    return JSONResponse({
        "predicted_label": predicted_label,
        "confidence": f"{confidence_score:.2f}%"
    })
