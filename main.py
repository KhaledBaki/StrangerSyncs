# Computer vision 
import cv2


def main():

    # Webcam index is 0
    camera = cv2.VideoCapture(0)

    # The case where camera is NOT found
    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return

    # The case where camera opened 
    print("Webcam opened. Click the video window and press Q to quit.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("ERROR: Could not read a frame from the webcam.")
                break

            cv2.imshow("Hack the Hill - Webcam Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Webcam closed.")


if __name__ == "__main__":
    main()