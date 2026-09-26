# Connects the program to the database

import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("personal-information-809ad-firebase-adminsdk-fbsvc-63f96afbdb.json")

firebase_admin.initialize_app(cred)

db = firestore.client()

print("Connected to Firebase!")