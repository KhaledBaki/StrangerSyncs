# Gives other files access to our one Firebase connection.

from .firebase_database import get_database

db = get_database()