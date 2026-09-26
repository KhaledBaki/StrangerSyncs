import cv2

from face_detection import (
    create_face_detector,
    detect_faces,
    draw_faces,
    get_largest_face,
)


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

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("ERROR: Could not read a frame from the webcam.")
                break

            faces = detect_faces(detector, frame)
            largest_face = get_largest_face(faces)
            draw_faces(frame, faces, largest_face)

            status = f"Faces detected: {len(faces)} | Press Q to quit"
            cv2.putText(
                frame,
                status,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Hack the Hill - Face Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Webcam closed.")


if __name__ == "__main__":
    main()