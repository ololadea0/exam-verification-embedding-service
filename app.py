from fastapi import FastAPI
from pydantic import BaseModel
from embedding import get_embedding_result

import cv2
import numpy as np
import base64

app = FastAPI()


class ImageRequest(BaseModel):
    images: list[str] | str   # one or more base64 images


def decode_base64(image_b64):
    img_data = base64.b64decode(image_b64.split(",")[-1])
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img


class SingleImageRequest(BaseModel):
    image: str


@app.post("/embed")
def embed_single(data: SingleImageRequest):

    try:
        image = decode_base64(data.image)

        if image is None:
            return {
                "error": "Invalid image data or format",
                "detail": "Could not decode or parse base64 image"
            }

        result = get_embedding_result(image)

        return {
            "embedding": result["embedding"],
            "metrics": result["metrics"]
        }
    except Exception as e:
        return {
            "error": str(e),
            "detail": "Face embedding generation failed"
        }
@app.post("/embed-best")
def embed_best(data: ImageRequest):

    image_list = data.images if isinstance(data.images, list) else [data.images]
    candidates = []

    for index, img_b64 in enumerate(image_list):

        try:
            image = decode_base64(img_b64)

            result = get_embedding_result(image)

            quality_score = (
                result["metrics"]["confidence"]
                + result["metrics"]["brightness_score"]
                + (result["metrics"]["sharpness"] / 100)
            )

            candidates.append({
                "index": index,
                "embedding": result["embedding"],
                "metrics": result["metrics"],
                "quality": quality_score
            })

        except Exception as e:
            candidates.append({
                "index": index,
                "error": str(e),
                "quality": 0
            })

    valid = [c for c in candidates if "embedding" in c]

    if not valid:
        return {
            "error": "No usable face detected",
            "candidates": candidates
        }

    best = max(valid, key=lambda c: c["quality"])

    return {
        "embedding": best["embedding"],
        "selectedIndex": best["index"],
        "candidates": candidates
    }