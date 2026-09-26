import time

import cv2

from database.database import get_all_people, increase_conversations
from face_detection import (
    create_face_detector,
    detect_faces,
    draw_faces,
    get_largest_face,
)
from face_recognition import find_best_match, get_face_embedding


WAITING_FOR_FACE = "WAITING_FOR_FACE"
WAITING_FOR_USER = "WAITING_FOR_USER"
WAITING_TO_LEAVE = "WAITING_TO_LEAVE"

STABLE_SECONDS = 0.8
LEAVE_SECONDS = 2.0


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

            elif state == WAITING_TO_LEAVE:
                if len(faces) == 0:
                    if absent_since is None:
                        absent_since = now

                    if now - absent_since >= LEAVE_SECONDS:
                        state = WAITING_FOR_FACE
                        stable_since = None
                        absent_since = None
                        message = "Ready for the next person"
                        print("Face left. Ready for the next person.")
                else:
                    absent_since = None

            draw_faces(frame, faces, largest_face)

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

            # Lock immediately: another frame must not trigger another update.
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
                    message = "Not found - step out to reset"
                    print("Person not found.")
                    print("Adding unknown people comes in Stage 7.")
                    continue

                new_count = increase_conversations(person["id"])

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
                print("Step out of view to reset; do not retry yet.")

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Webcam closed.")


if __name__ == "__main__":
    main()