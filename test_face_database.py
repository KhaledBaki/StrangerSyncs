import cv2

from database.database import create_person, get_all_people
from face_detection import (
    create_face_detector,
    detect_faces,
    draw_faces,
    get_largest_face,
)
from face_recognition import find_best_match, get_face_embedding


def capture_face():
    detector = create_face_detector()
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return None

    print("Face the camera. Press S to capture the green-boxed face.")
    print("Press Q to cancel.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("ERROR: Could not read from webcam.")
                return None

            faces = detect_faces(detector, frame)
            largest_face = get_largest_face(faces)

            clean_frame = frame.copy()
            draw_faces(frame, faces, largest_face)

            cv2.putText(
                frame,
                "S: capture | Q: quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Face Database Test", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                return None

            if key == ord("s"):
                if largest_face is None:
                    print("No face detected. Try again.")
                    continue

                return clean_frame, largest_face
    finally:
        camera.release()
        cv2.destroyAllWindows()


def ask_for_details():
    name = input("Name: ").strip()
    while not name:
        name = input("Name cannot be empty. Name: ").strip()

    while True:
        age_text = input("Age: ").strip()
        if age_text.isdigit() and 0 < int(age_text) < 120:
            age = int(age_text)
            break
        print("Enter an age from 1 to 119.")

    phone = input("Phone: ").strip()
    while not phone:
        phone = input("Phone cannot be empty. Phone: ").strip()

    return name, age, phone


def main():
    captured = capture_face()

    if captured is None:
        print("Cancelled.")
        return

    frame, face = captured

    try:
        print("Generating face embedding...")
        embedding = get_face_embedding(frame, face)

        print("Searching Firestore...")
        people = get_all_people()
        person, distance = find_best_match(embedding, people)

        if person is not None:
            print("\nPERSON FOUND")
            print(f"Name: {person['name']}")
            print(f"Age: {person['age']}")
            print(f"Phone: {person['phone']}")
            print(f"Conversations: {person['conversations']}")
            print(f"Cosine distance: {distance:.3f}")
            print("Conversation count was NOT changed in this test.")
            return

        print("\nNo matching person found.")
        if distance is not None:
            print(f"Closest distance: {distance:.3f}")
        print(f"Match threshold: 0.40")

        answer = input(
            "Does this person consent to being added? (Y/N): "
        ).strip().lower()

        if answer != "y":
            print("Nothing saved.")
            return

        name, age, phone = ask_for_details()
        person_id = create_person(name, age, phone, embedding)

        print("\nPERSON ADDED")
        print(f"ID: {person_id}")
        print(f"Name: {name}")
        print(f"Age: {age}")
        print(f"Phone: {phone}")
        print("Conversations: 1")

    except Exception as error:
        print(f"ERROR: Face database test failed: {error}")


if __name__ == "__main__":
    main()