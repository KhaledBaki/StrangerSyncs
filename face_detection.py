import cv2


def create_face_detector():
    cascade_path = (
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    detector = cv2.CascadeClassifier(cascade_path)

    if detector.empty():
        raise RuntimeError("Could not load the OpenCV face detector.")

    return detector


def detect_faces(detector, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )


def get_largest_face(faces):
    if len(faces) == 0:
        return None

    return max(faces, key=lambda face: int(face[2]) * int(face[3]))


def draw_faces(frame, faces, largest_face):
    # Show other detected faces in gray.
    for x, y, width, height in faces:
        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            (160, 160, 160),
            2,
        )

    # Highlight only the selected face in green.
    if largest_face is not None:
        x, y, width, height = largest_face

        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            3,
        )

        cv2.putText(
            frame,
            "Closest face",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )