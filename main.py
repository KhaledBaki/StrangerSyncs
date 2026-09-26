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
    draw_entry_form,
    draw_faces,
    draw_person_card,
    find_tracked_face,
    get_largest_face,
)
from face_recognition import find_best_match, get_face_embedding


WAITING_FOR_FACE = "WAITING_FOR_FACE"
WAITING_FOR_USER = "WAITING_FOR_USER"
ASKING_TO_ADD = "ASKING_TO_ADD"
ENTERING_DETAILS = "ENTERING_DETAILS"
WAITING_TO_LEAVE = "WAITING_TO_LEAVE"

STABLE_SECONDS = 0.8
LEAVE_SECONDS = 2.0
WINDOW_NAME = "StrangerSyncs"
FIELDS = ("name", "age", "phone")


def validate_field(field, value):
    value = value.strip()

    if field == "name":
        return None if value else "Enter a name."

    if field == "age":
        if value.isdigit() and 0 < int(value) < 120:
            return None
        return "Enter an age from 1 to 119."

    digits = [character for character in value if character.isdigit()]
    allowed = all(
        character in "0123456789+ -()."
        for character in value
    )

    if allowed and 7 <= len(digits) <= 15:
        return None

    return "Phone needs 7 to 15 digits."


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
    form_face = None

    form_values = {"name": "", "age": "", "phone": ""}
    field_index = 0
    form_error = ""

    message = "Show a face to the camera"
    window_created = False

    print("Camera controls: S search, N ignore, Q quit.")
    print("Press F to toggle full screen.")

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

            elif state == ENTERING_DETAILS and form_face is not None:
                current_face = find_tracked_face(faces, form_face)

                if current_face is not None:
                    form_face = current_face

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
                f"Faces: {len(faces)} | Q: quit | F: full screen",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
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

            if state == ENTERING_DETAILS:
                draw_entry_form(
                    frame, form_values, field_index, form_error
                )

            if not window_created:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, frame)
                window_created = True

                try:
                    cv2.setWindowProperty(
                        WINDOW_NAME,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN,
                    )
                    fullscreen = True
                except cv2.error:
                    fullscreen = False
                    print("Full screen unavailable; using a normal window.")
            else:
                cv2.imshow(WINDOW_NAME, frame)

            raw_key = cv2.waitKey(1)
            key = raw_key & 0xFF if raw_key != -1 else -1

            # While typing, Q and F are ordinary letters, not controls.
            if state == ENTERING_DETAILS:
                if key == 27:  # Esc
                    pending_embedding = None
                    form_face = None
                    state = WAITING_TO_LEAVE
                    absent_since = None
                    message = "Add cancelled - step out"
                    continue

                if key in (8, 127):  # Backspace
                    field = FIELDS[field_index]

                    if form_values[field]:
                        form_values[field] = form_values[field][:-1]
                    elif field_index > 0:
                        field_index -= 1

                    form_error = ""
                    continue

                if key in (10, 13):  # Enter
                    field = FIELDS[field_index]
                    error = validate_field(
                        field, form_values[field]
                    )

                    if error is not None:
                        form_error = error
                        continue

                    form_values[field] = form_values[field].strip()
                    form_error = ""

                    if field_index < len(FIELDS) - 1:
                        field_index += 1
                        continue

                    # Last field validated: save exactly once.
                    name = form_values["name"]
                    age = int(form_values["age"])
                    phone = form_values["phone"]
                    embedding_to_save = pending_embedding

                    pending_embedding = None
                    state = WAITING_TO_LEAVE
                    absent_since = None
                    message = "Saving person..."

                    try:
                        person_id = create_person(
                            name, age, phone, embedding_to_save
                        )

                        active_person = {
                            "name": name,
                            "age": age,
                            "phone": phone,
                            "conversations": 1,
                        }
                        tracked_face = form_face
                        form_face = None

                        message = f"Added {name} - step out"
                        print("\nPERSON ADDED")
                        print(f"ID: {person_id}")
                        print(f"Name: {name}")
                        print(f"Age: {age}")
                        print(f"Phone: {phone}")
                        print("Conversations: 1")
                        print("Step out of view to reset.\n")

                    except Exception as error:
                        form_face = None
                        message = "Save failed - see terminal"
                        print(f"ERROR: Could not add person: {error}")
                        print(
                            "Check Firestore before trying again "
                            "to avoid duplicates."
                        )

                    continue

                if 32 <= key <= 126:
                    field = FIELDS[field_index]
                    character = chr(key)

                    if field == "age" and not character.isdigit():
                        form_error = "Age: numbers only."
                        continue

                    if (
                        field == "phone"
                        and character not in "0123456789+ -()."
                    ):
                        form_error = "Phone: use digits and + - ( ) spaces."
                        continue

                    max_length = {
                        "name": 40,
                        "age": 3,
                        "phone": 24,
                    }[field]

                    if len(form_values[field]) >= max_length:
                        form_error = "Field is too long."
                        continue

                    form_values[field] += character
                    form_error = ""

                continue

            if key in (ord("q"), ord("Q"), 27):
                break

            if key in (ord("f"), ord("F")):
                try:
                    fullscreen = not fullscreen
                    cv2.setWindowProperty(
                        WINDOW_NAME,
                        cv2.WND_PROP_FULLSCREEN,
                        (
                            cv2.WINDOW_FULLSCREEN
                            if fullscreen
                            else cv2.WINDOW_NORMAL
                        ),
                    )
                except cv2.error as error:
                    print(f"Could not change window mode: {error}")
                continue

            if state == ASKING_TO_ADD:
                if key in (ord("n"), ord("N")):
                    pending_embedding = None
                    state = WAITING_TO_LEAVE
                    absent_since = None
                    message = "Skipped - step out to reset"
                    print("Nothing saved. Step out to reset.")

                elif key in (ord("y"), ord("Y")):
                    state = ENTERING_DETAILS
                    form_face = largest_face
                    form_values = {
                        "name": "",
                        "age": "",
                        "phone": "",
                    }
                    field_index = 0
                    form_error = ""
                    message = "Enter details on screen"
                    print("Entering details in the camera window.")

                continue

            if state != WAITING_FOR_USER:
                continue

            if key in (ord("n"), ord("N")):
                state = WAITING_TO_LEAVE
                absent_since = None
                message = "Ignored - step out to reset"
                print("Person ignored.")
                continue

            if key not in (ord("s"), ord("S")):
                continue

            # Lock before recognition or any Firebase operation.
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
                    print("Person not found. Press Y or N in the window.")
                    continue

                new_count = increase_conversations(person["id"])

                active_person = {
                    "name": person["name"],
                    "age": person["age"],
                    "phone": person["phone"],
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