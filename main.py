import cv2

from face_detection import (
    create_face_detector,
    detect_faces,
    draw_faces,
    get_largest_face,
)
from face_recognition import get_face_embedding


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

    print("Webcam opened. Click the video window and press Q to quit.")
    message = "Show your face, then press S"

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("ERROR: Could not read a frame from the webcam.")
                break

            faces = detect_faces(detector, frame)
            largest_face = get_largest_face(faces)

            # Keep an untouched frame for DeepFace. Drawing on it first
            # would put the green box into the image we analyze.
            clean_frame = frame.copy()

            draw_faces(frame, faces, largest_face)

            cv2.putText(
                frame,
                f"Faces: {len(faces)} | S: embedding | Q: quit",
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

            cv2.imshow("Hack the Hill - Face Embedding Test", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("s"):
                if largest_face is None:
                    message = "No face detected - try again"
                    print(message)
                    continue

                message = "Generating embedding..."
                print(message)

                try:
                    embedding = get_face_embedding(
                        clean_frame, largest_face
                    )
                    message = f"Success: {len(embedding)} numbers"
                    print(message)
                except Exception as error:
                    message = "Embedding failed - see terminal"
                    print(f"ERROR: Could not generate embedding: {error}")

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Webcam closed.")


if __name__ == "__main__":
    main()