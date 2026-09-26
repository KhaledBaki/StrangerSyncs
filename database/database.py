import uuid

from firebase_admin import firestore

from face_recognition import MODEL_NAME
from .firebase_config import db


def create_person(name, age, phone, face_embedding):
    person_id = str(uuid.uuid4())

    person_data = {
        "name": name,
        "age": age,
        "phone": phone,
        "conversations": 1,
        "face_embedding": face_embedding,
        "face_model": MODEL_NAME,
    }

    db.collection("people").document(person_id).set(person_data)

    print(f"Person created! ID: {person_id}")
    return person_id


def get_all_people():
    people = []

    for document in db.collection("people").stream():
        person = document.to_dict()
        person["id"] = document.id
        people.append(person)

    return people


def increase_conversations(person_id):
    document = db.collection("people").document(person_id)

    document.update({
        "conversations": firestore.Increment(1),
    })

    updated = document.get()

    if not updated.exists:
        raise RuntimeError("Person disappeared after the update.")

    return updated.to_dict()["conversations"]