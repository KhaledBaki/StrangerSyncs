import uuid

from .firebase_config import db


def create_person(name, age, phone, face_embedding):
    person_id = str(uuid.uuid4())

    person_data = {
        "name": name,
        "age": age,
        "phone": phone,
        "conversations": 1,
        "face_embedding": face_embedding,
    }

    db.collection("people").document(person_id).set(person_data)

    print(f"Person created!")
    print(f"Person ID: {person_ID}")

    return person_ID



