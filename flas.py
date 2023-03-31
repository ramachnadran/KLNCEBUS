import cv2
import numpy as np
from flask import Flask, Response

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

app = Flask(__name__)

cap = cv2.VideoCapture(0)

@app.route('/')
def index():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    while True:
        ret, frame = cap.read()

        # Check if camera is off
        if not ret:
            yield b'--frame\r\n' b'Content-Type: text/plain\r\n\r\n' b'Hello Ramar! Camera is off.\r\n'
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        heads = face_cascade.detectMultiScale(gray, 1.1, 4)

        num_heads = len(heads)

        if num_heads == 0:
            color = (255, 255, 255)  # white
            status = "Bus not started"

        elif num_heads <= 50:
            color = (0, 255, 0)  # green
            status = "Available seats in bus"

        elif num_heads >= 50 and num_heads <= 60:

            color = (0, 165, 255)  # orange
            status = "Available place in standing"

        else:
            color = (0, 0, 255)  # red
            status = "No space in bus"


        # Create a new blank image
        status_image = np.zeros((100, 640, 3), np.uint8)

        # Draw the color rectangle at the top
        cv2.rectangle(status_image, (0, 0), (640, 50), color, -1)

        # Draw the status text at the bottom
        cv2.putText(status_image, status, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', status_image)
        status_image = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + status_image + b'\r\n')

if __name__ == '__main__':
    app.run(debug=True)
