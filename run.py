from ultralytics import YOLO
import cv2
import cvzone
import math
import serial
import time

# Initialize serial connection
ser = serial.Serial("/dev/cu.usbserial-1330",9600)# Replace 'COM3' with your Arduino port

def draw_boxes_labels(img, results, classnames):
    global ser
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            letter = classnames[cls]
            cvzone.putTextRect(img, f'{letter} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
            # Send recognized letter to Arduino
            ser.write(letter.encode())  # Sending the letter to Arduino
            time.sleep(1)  # Delay to avoid flooding the serial port

cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)
model = YOLO('/Users/ryanoliver/PycharmProjects/AS_L/runs/detect/train4/weights/best.pt')

classnames = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

while True:
    success, img = cap.read()
    if not success or img is None:
        print("Error capturing frame from the webcam.")
        continue

    results = model(img)
    draw_boxes_labels(img, results, classnames)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break

# Close the serial connection when done
ser.close()
cv2.destroyAllWindows()