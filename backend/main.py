import io
import json
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "harvestguard.keras"
METADATA_PATH = MODEL_DIR / "metadata.json"

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    raise RuntimeError(
        "Copy harvestguard.keras and metadata.json "
        "into backend/models before starting the server."
    )

with METADATA_PATH.open(encoding="utf-8") as file:
    metadata = json.load(file)

class_names = metadata["class_names"]
image_size = tuple(metadata["image_size"])

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

if model.output_shape[-1] != len(class_names):
    raise RuntimeError(
        "The model outputs do not match metadata.json."
    )


app = FastAPI(title="HarvestGuard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": metadata.get(
            "model_name",
            "HarvestGuard CNN"
        ),
        "number_of_classes": len(class_names)
    }


def readable_name(class_name):
    return class_name.replace(
        "___",
        ": "
    ).replace(
        "_",
        " "
    )


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    contents = file.file.read(MAX_UPLOAD_BYTES + 1)

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Please upload an image."
        )

    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Please choose an image smaller than 10 MB."
        )

    try:
        with Image.open(io.BytesIO(contents)) as original:
            if original.width * original.height > 25_000_000:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Please upload an image below "
                        "25 megapixels."
                    )
                )

            image = ImageOps.exif_transpose(
                original
            ).convert("RGB")

            array = np.asarray(
                image,
                dtype=np.float32
            )

    except HTTPException:
        raise

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError
    ):
        raise HTTPException(
            status_code=400,
            detail="That file could not be read as an image."
        )

    try:
        # The saved MobileNetV2 model contains its own
        # preprocessing layer and expects 0–255 RGB pixels.
        resized = tf.image.resize(
            array,
            image_size
        )

        batch = tf.expand_dims(
            resized,
            axis=0
        )

        scores = model.predict(
            batch,
            verbose=0
        )[0]

        if not np.all(np.isfinite(scores)):
            raise ValueError(
                "Model returned invalid scores."
            )

        predictions = [
            {
                "class_name": readable_name(name),
                "score": float(score)
            }
            for name, score in zip(
                class_names,
                scores
            )
        ]

        predictions.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        return {
            "prediction": predictions[0]["class_name"],
            "model_score": predictions[0]["score"],
            "scores": predictions,
            "notice": (
                "This model supports only the 38 trained "
                "PlantVillage crop-disease and healthy classes. "
                "Scores do not guarantee correctness."
            )
        }

    except Exception:
        logging.exception(
            "Image prediction failed."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction failed. Check the backend "
                "terminal."
            )
        )