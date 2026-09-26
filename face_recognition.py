import numpy as np
from deepface import DeepFace


MODEL_NAME = "Facenet"


def get_face_embedding(frame, face):
    x, y, width, height = [int(value) for value in face]

    # Keep a little space around the detected face.
    padding_x = int(width * 0.15)
    padding_y = int(height * 0.15)

    left = max(0, x - padding_x)
    top = max(0, y - padding_y)
    right = min(frame.shape[1], x + width + padding_x)
    bottom = min(frame.shape[0], y + height + padding_y)

    face_image = frame[top:bottom, left:right]

    if face_image.size == 0:
        raise ValueError("The selected face image was empty.")

    results = DeepFace.represent(
        img_path=face_image,
        model_name=MODEL_NAME,
        detector_backend="skip",
    )

    embedding = np.asarray(results[0]["embedding"], dtype=np.float64)

    if embedding.size == 0 or not np.all(np.isfinite(embedding)):
        raise ValueError("DeepFace returned an invalid embedding.")

    return embedding.tolist()