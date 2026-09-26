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
    for x, y, width, height in faces:
        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            (160, 160, 160),
            2,
        )

    if largest_face is not None:
        x, y, width, height = largest_face

        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            (0, 165, 255),
            3,
        )


def find_tracked_face(faces, previous_face):
    """Find a face box that overlaps the previously recognized box."""
    px, py, pw, ph = [int(value) for value in previous_face]
    best_face = None
    best_overlap = 0.0

    for face in faces:
        x, y, width, height = [int(value) for value in face]

        left = max(px, x)
        top = max(py, y)
        right = min(px + pw, x + width)
        bottom = min(py + ph, y + height)

        intersection = max(0, right - left) * max(0, bottom - top)
        union = pw * ph + width * height - intersection
        overlap = intersection / union if union else 0.0

        if overlap > best_overlap:
            best_overlap = overlap
            best_face = face

    # If no box is close enough, hide the card rather than
    # putting this person's details above a different face.
    return best_face if best_overlap >= 0.10 else None


def draw_person_card(frame, face, person):
    x, y, width, height = [int(value) for value in face]
    frame_height, frame_width = frame.shape[:2]

    lines = [
        f"Name: {person['name']}",
        f"Age: {person['age']}",
        f"Conversations: {person['conversations']}",
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    text_thickness = 1

    text_width = max(
        cv2.getTextSize(
            line, font, font_scale, text_thickness
        )[0][0]
        for line in lines
    )

    card_width = min(frame_width - 16, max(180, text_width + 20))
    card_height = 84

    # Keep the card inside the camera image horizontally.
    card_x = max(8, min(x, frame_width - card_width - 8))

    # Prefer above the face; use below if there isn't enough room.
    if y - card_height - 8 >= 8:
        card_y = y - card_height - 8
    elif y + height + 8 + card_height <= frame_height:
        card_y = y + height + 8
    else:
        card_y = max(8, min(y, frame_height - card_height - 8))

    cv2.rectangle(
        frame,
        (card_x, card_y),
        (card_x + card_width, card_y + card_height),
        (255, 0, 0),
        -1,
    )

    for index, line in enumerate(lines):
        # Shorten unusually long names so text stays inside the box.
        displayed = line
        while (
            len(displayed) > 3
            and cv2.getTextSize(
                displayed, font, font_scale, text_thickness
            )[0][0] > card_width - 16
        ):
            displayed = displayed[:-4] + "..."

        cv2.putText(
            frame,
            displayed,
            (card_x + 8, card_y + 21 + index * 23),
            font,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )