import numpy as np
from deepface import DeepFace


MODEL_NAME = "Facenet"
MATCH_THRESHOLD = 0.40


def get_face_embedding(frame, face):
    x, y, width, height = [int(value) for value in face]

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


def find_best_match(embedding, people):
    query = np.asarray(embedding, dtype=np.float64)
    query_norm = np.linalg.norm(query)

    if query_norm == 0:
        raise ValueError("Cannot compare an empty face embedding.")

    best_person = None
    best_distance = None

    for person in people:
        stored = person.get("face_embedding")

        if person.get("face_model") != MODEL_NAME:
            continue

        if not isinstance(stored, list) or len(stored) != len(query):
            continue

        saved = np.asarray(stored, dtype=np.float64)
        saved_norm = np.linalg.norm(saved)

        if saved_norm == 0 or not np.all(np.isfinite(saved)):
            continue

        similarity = np.dot(query, saved) / (query_norm * saved_norm)
        distance = float(1 - similarity)

        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_person = person

    if best_distance is not None and best_distance <= MATCH_THRESHOLD:
        return best_person, best_distance

    return None, best_distance