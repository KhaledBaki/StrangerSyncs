# methods for adding/deleting data person

# generates a unique ID for each person
import uuid

from firebase_config import db

# Create a person and store it in the database of collection "people"
def create_person(name, age, phone):
    person_ID = str(uuid.uuid4()); 

    person_data = {
        "name": name, 
        "age": age, 
        "phone":  phone,
    }

    db.collection("people").document(person_ID).set(person_data)

    print(f"Person created!")
    print(f"Person ID: {person_ID}")

    return person_ID



