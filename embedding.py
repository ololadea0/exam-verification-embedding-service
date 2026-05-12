import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np
from deepface import DeepFace

ARC_FACE_MODEL = DeepFace.build_model("ArcFace")



MIN_DETECTION_CONFIDENCE = 0.4
MIN_FACE_AREA_RATIO = 0.008
MAX_FACE_AREA_RATIO = 0.8
MIN_SHARPNESS = 5
MIN_BRIGHTNESS_SCORE = 0.12


# -------------------------------
# IMAGE QUALITY METRICS
# -------------------------------
def calculate_image_metrics(image, face):

    if image is None:
        raise ValueError("Image is empty or invalid")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    brightness = float(np.mean(gray))
    brightness_score = max(0, 1 - abs(brightness - 128) / 128)

    image_height, image_width = image.shape[:2]

    facial_area = face.get("facial_area", {})

    face_width = facial_area.get("w", 0)
    face_height = facial_area.get("h", 0)

    face_area_ratio = (
        (face_width * face_height)
        / max(image_width * image_height, 1)
    )

    confidence = float(
        face.get("confidence", face.get("face_confidence", 1))
    )

    return {
        "confidence": confidence,
        "face_area_ratio": face_area_ratio,
        "sharpness": sharpness,
        "brightness_score": brightness_score
    }


# -------------------------------
# VALIDATION RULES
# -------------------------------
def validate_metrics(metrics):

    if metrics["confidence"] < MIN_DETECTION_CONFIDENCE:
        raise ValueError("Face detection confidence too low")

    if metrics["face_area_ratio"] < MIN_FACE_AREA_RATIO:
        raise ValueError("Face too small")

    if metrics["face_area_ratio"] > MAX_FACE_AREA_RATIO:
        raise ValueError("Face too close")

    if metrics["sharpness"] < MIN_SHARPNESS:
        raise ValueError("Image too blurry")

    # if metrics["brightness_score"] < MIN_BRIGHTNESS_SCORE:
    #     raise ValueError("Poor lighting")


# -------------------------------
# CORE EMBEDDING FUNCTION (NEW)
# -------------------------------
def get_embedding_result(image):

    """
    image = numpy array (NOT file path anymore)
    """
  

    faces = DeepFace.extract_faces(
    img_path=image,
    enforce_detection=True,
    detector_backend="opencv",
    align=True
)

    if not faces:
        raise ValueError("No face detected")

    best_face = max(
        faces,
        key=lambda x: x.get("confidence", x.get("face_confidence", 1))
    )

    metrics = calculate_image_metrics(image, best_face)

    validate_metrics(metrics)

    result = DeepFace.represent(
        img_path=image,
        model_name="ArcFace",
        model=ARC_FACE_MODEL,
        enforce_detection=True,
        detector_backend="opencv",
        align=True
    )

    return {
        "embedding": result[0]["embedding"],
        "metrics": metrics
    }



def normalize(embedding):
    arr = np.array(embedding)
    return (arr / np.linalg.norm(arr)).tolist()


# -------------------------------
# SIMPLE WRAPPER
# -------------------------------
def get_embedding(image):
    return normalize(get_embedding_result(image)["embedding"])