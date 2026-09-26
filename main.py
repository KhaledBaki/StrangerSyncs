import time

import cv2

from database.database import (
    create_person,
    get_all_people,
    increase_conversations,
)
from face_detection import (
    create_face_detector,
    detect_faces,
    draw_faces,
    draw_person_card,
    find_tracked_face,
    get_largest_face,
)
from face_recognition import find_best_match, get_face_embedding


WAITING_FOR_FACE = "WAITING_FOR_FACE"
WAITING_FOR_USER = "WAITING_FOR_USER"
ASKING_TO_ADD = "ASKING_TO_ADD"
WAITING_TO_LEAVE = "WAITING_TO_LEAVE"

STABLE_SECONDS = 0.8
LEAVE_SECONDS = 2.0


def ask_for_details():
    print("\nEnter the person's details in this terminal.")

    try:
        while True:
            name = input("Name: ").strip()
            if name:
                break
            print("Name cannot be empty.")

        while True:
            age_text = input("Age: ").strip()
            if age_text.isdigit() and 0 < int(age_text) < 120:
                age = int(age_text)
                break
            print("Enter an age from 1 to 119.")

        while True:
            phone = input("Phone: ").strip()
            digits = [character for character in phone if character.isdigit()]
            allowed = all(
                character in "0123456789+ -()."
                for character in phone
            )

            if allowed and 7 <= len(digits) <= 15:
                break

            print("Enter a phone number with 7 to 15 digits.")

        return name, age, phone

    except (EOFError, KeyboardInterrupt):
        print("\nAdding person cancelled.")
        return None


def main():
    try:
        detector = create_face_detector()
    except RuntimeError as error:
        print(f"ERROR: {error}")
        return

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return

    state = WAITING_FOR_FACE
    stable_since = None
    absent_since = None
    pending_embedding = None

    active_person = None
    tracked_face = None

    message = "Show a face to the camera"
    print("Press Q in the camera window to quit.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("ERROR: Could not read from webcam.")
                break

            faces = detect_faces(detector, frame)
            largest_face = get_largest_face(faces)
            clean_frame = frame.copy()
            now = time.monotonic()

            if state == WAITING_FOR_FACE:
                if largest_face is None:
                    stable_since = None
                    message = "Show a face to the camera"
                else:
                    if stable_since is None:
                        stable_since = now

                    if now - stable_since >= STABLE_SECONDS:
                        state = WAITING_FOR_USER
                        message = "S: search | N: ignore"

            elif state == WAITING_FOR_USER:
                if largest_face is None:
                    state = WAITING_FOR_FACE
                    stable_since = None
                    message = "Face lost - try again"

            elif state == ASKING_TO_ADD:
                if len(faces) == 0:
                    if absent_since is None:
                        absent_since = now

                    if now - absent_since >= LEAVE_SECONDS:
                        pending_embedding = None
                        state = WAITING_FOR_FACE
                        stable_since = None
                        absent_since = None
                        message = "Face left - add cancelled"
                        print("Face left. Nothing was saved.")
                else:
                    absent_since = None

            elif state == WAITING_TO_LEAVE:
                if len(faces) == 0:
                    if absent_since is None:
                        absent_since = now

                    if now - absent_since >= LEAVE_SECONDS:
                        state = WAITING_FOR_FACE
                        stable_since = None
                        absent_since = None
                        active_person = None
                        tracked_face = None
                        message = "Ready for the next person"
                        print("Face left. Ready for the next person.")
                else:
                    absent_since = None

            draw_faces(frame, faces, largest_face)

            if active_person is not None and tracked_face is not None:
                current_face = find_tracked_face(
                    faces, tracked_face
                )

                if current_face is not None:
                    tracked_face = current_face
                    draw_person_card(
                        frame, current_face, active_person
                    )

            cv2.putText(
                frame,
                f"Faces: {len(faces)} | Q: quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                message,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow("StrangerSyncs", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if state == ASKING_TO_ADD:
                if key == ord("n"):
                    pending_embedding = None
                    state = WAITING_TO_LEAVE
                    absent_since = None
                    message = "Skipped - step out to reset"
                    print("Nothing saved. Step out to reset.")

                elif key == ord("y"):
                    embedding_to_save = pending_embedding
                    pending_embedding = None
                    state = WAITING_TO_LEAVE
                    absent_since = None
                    message = "Enter details in PowerShell"

                    print("Click the PowerShell terminal to type.")
                    details = ask_for_details()

                    if details is None:
                        message = "Add cancelled - step out"
                        continue

                    name, age, phone = details

                    try:
                        person_id = create_person(
                            name, age, phone, embedding_to_save
                        )

                        active_person = {
                            "name": name,
                            "age": age,
                            "conversations": 1,
                        }
                        tracked_face = largest_face

                        message = f"Added {name} - step out"
                        print("\nPERSON ADDED")
                        print(f"ID: {person_id}")
                        print(f"Name: {name}")
                        print(f"Age: {age}")
                        print(f"Phone: {phone}")
                        print("Conversations: 1")
                        print("Step out of view to reset.\n")

                    except Exception as error:
                        message = "Save failed - see terminal"
                        print(f"ERROR: Could not add person: {error}")
                        print("Step out to reset. Check Firestore before retrying.")

                continue

            if state != WAITING_FOR_USER:
                continue

            if key == ord("n"):
                state = WAITING_TO_LEAVE
                absent_since = None
                message = "Ignored - step out to reset"
                print("Person ignored.")
                continue

            if key != ord("s"):
                continue

            state = WAITING_TO_LEAVE
            absent_since = None
            message = "Searching - see terminal"
            print("Generating embedding and searching Firestore...")

            try:
                embedding = get_face_embedding(
                    clean_frame, largest_face
                )
                people = get_all_people()
                person, distance = find_best_match(
                    embedding, people
                )

                if person is None:
                    pending_embedding = embedding
                    state = ASKING_TO_ADD
                    message = "Not found: Y add | N skip"
                    print("Person not found. Press Y or N in the camera window.")
                    continue

                new_count = increase_conversations(person["id"])

                active_person = {
                    "name": person["name"],
                    "age": person["age"],
                    "conversations": new_count,
                }
                tracked_face = largest_face

                message = (
                    f"{person['name']}: conversations {new_count}"
                )

                print("\nPERSON FOUND")
                print(f"Name: {person['name']}")
                print(f"Age: {person['age']}")
                print(f"Phone: {person['phone']}")
                print(f"Conversations: {new_count}")
                print(f"Cosine distance: {distance:.3f}")
                print("Step out of view to reset.\n")

            except Exception as error:
                message = "Search failed - see terminal"
                print(f"ERROR: Could not process person: {error}")
                print("Step out to reset; do not retry yet.")

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Webcam closed.")


if __name__ == "__main__":
    main()