from ultralytics import YOLO
import cv2
import cvzone
import math
from gtts import gTTS
from playsound import playsound
import time

def draw_boxes_labels(img, results, classnames):
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            letter = classnames[cls]
            detection_text = f"Detected: {letter} with confidence: {conf}"
            print(detection_text)
            cvzone.putTextRect(img, f'{letter} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
            # Convert text to speech using gTTS and play it with a delay
            tts = gTTS(text=detection_text, lang='en')
            tts.save("output.mp3")  # Save as temporary file
            time.sleep(2)  # Adjust the delay time (in seconds) as needed
            playsound("output.mp3", False)  # Play audio without blocking

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
model = YOLO('/Users/ryanoliver/PycharmProjects/AS_L/runs/detect/train4/weights/best.pt')

classnames = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

while True:
    success, img = cap.read()
    if not success or img is None:
        print("Error capturing frame from the webcam.")
        continue

    results = model(img, stream=True)
    draw_boxes_labels(img, results, classnames)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
