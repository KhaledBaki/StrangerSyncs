import os
from pathlib import Path

import firebase_admin
from firebase_admin import firestore


def get_database():
    key_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

    if not key_path or not Path(key_path).is_file():
        raise FileNotFoundError(
            "Firebase key not found. Set GOOGLE_APPLICATION_CREDENTIALS "
            "to your service account JSON file."
        )

    try:
        firebase_admin.get_app()
    except ValueError:
        firebase_admin.initialize_app()

    return firestore.client()