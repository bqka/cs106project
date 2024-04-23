from modelutils import *
import cv2
from ultralytics import YOLO

def get_faces(image):
    yolo_model = YOLO('yolov8.pt')
    result = yolo_model(image)
    boxes = result[0].boxes
    faces = []
    
    for box in boxes:
        top_left_x, top_left_y = int(box.xyxy.tolist()[0][0]), int(box.xyxy.tolist()[0][1])
        bottom_right_x, bottom_right_y = int(box.xyxy.tolist()[0][2]), int(box.xyxy.tolist()[0][3])

        faces.append([top_left_x, top_left_y, bottom_right_x, bottom_right_y])
    
    return faces

model = load('training_02')

# vid = int(input("Webcam?"))
vid = 0

if vid == 1:
    cap = cv2.VideoCapture(0)
else:
    cap = 'demo/anshul2.jpeg'

while True:
    if vid == 1:
        ret, image = cap.read()
    else:
        image = cv2.imread(cap)
    # image = cv2.resize(image, (0, 0), fx=0.2, fy=0.2)

    faces = get_faces(image)

    for top_left_x, top_left_y, bottom_right_x, bottom_right_y in faces:

        face = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (48, 48))
        
        emotion = predict(model, face)

        cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0,0,255), 3)
        cv2.putText(image, str(emotion), (top_left_x, top_left_y), cv2.FONT_HERSHEY_COMPLEX, 1.3, (255, 0, 0), 2)

    cv2.imshow("Output", image)
    if cv2.waitKey(1) == 27:
        break