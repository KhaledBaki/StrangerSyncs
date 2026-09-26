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
        x, y, width, height = [int(value) for value in largest_face]

        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            3,
        )


def find_tracked_face(faces, previous_face):
    """Find a detected box that overlaps the last known face box."""
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

    return best_face if best_overlap >= 0.10 else None


def fit_text(text, max_width, font_scale=0.55):
    """Shorten text only if it cannot fit inside its box."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    displayed = str(text)

    while (
        len(displayed) > 3
        and cv2.getTextSize(displayed, font, font_scale, 1)[0][0]
        > max_width
    ):
        displayed = displayed[:-4] + "..."

    return displayed


def draw_person_card(frame, face, person):
    x, y, width, height = [int(value) for value in face]
    frame_height, frame_width = frame.shape[:2]

    lines = [
        f"Name: {person['name']}",
        f"Age: {person['age']}",
        f"Phone: {person['phone']}",
        f"Conversations: {person['conversations']}",
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55

    text_width = max(
        cv2.getTextSize(line, font, font_scale, 1)[0][0]
        for line in lines
    )

    card_width = min(frame_width - 16, max(220, text_width + 20))
    card_height = 108

    card_x = max(8, min(x, frame_width - card_width - 8))

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
        cv2.putText(
            frame,
            fit_text(line, card_width - 16, font_scale),
            (card_x + 8, card_y + 22 + index * 23),
            font,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def draw_entry_form(frame, values, field_index, error):
    """Draw the new-person form over the live camera image."""
    frame_height, frame_width = frame.shape[:2]
    panel_width = min(frame_width - 20, 620)
    panel_height = 300

    left = (frame_width - panel_width) // 2
    top = max(10, (frame_height - panel_height) // 2)
    right = left + panel_width
    bottom = top + panel_height

    cv2.rectangle(
        frame, (left, top), (right, bottom), (35, 35, 35), -1
    )
    cv2.rectangle(
        frame, (left, top), (right, bottom), (255, 0, 0), 2
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(
        frame,
        "ADD NEW PERSON",
        (left + 15, top + 32),
        font,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        "Type here - Enter: next/save | Esc: cancel",
        (left + 15, top + 58),
        font,
        0.45,
        (210, 210, 210),
        1,
    )

    fields = ("name", "age", "phone")

    for index, field in enumerate(fields):
        row_top = top + 74 + index * 53
        selected = index == field_index

        cv2.rectangle(
            frame,
            (left + 12, row_top),
            (right - 12, row_top + 43),
            (130, 65, 25) if selected else (60, 60, 60),
            -1,
        )

        label = field.capitalize() + ":"
        value = values[field]

        cv2.putText(
            frame,
            label,
            (left + 22, row_top + 28),
            font,
            0.55,
            (255, 255, 255),
            1,
        )

        available_width = max(30, panel_width - 135)
        shown_value = fit_text(value, available_width, 0.55)

        cv2.putText(
            frame,
            shown_value,
            (left + 115, row_top + 28),
            font,
            0.55,
            (255, 255, 255),
            1,
        )

    cv2.putText(
        frame,
        fit_text(error, panel_width - 30, 0.48),
        (left + 15, top + 262),
        font,
        0.48,
        (100, 180, 255),
        1,
    )

    cv2.putText(
        frame,
        "Backspace: delete | Esc: cancel",
        (left + 15, top + 284),
        font,
        0.43,
        (200, 200, 200),
        1,
    )