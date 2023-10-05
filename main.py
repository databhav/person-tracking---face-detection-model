from ultralytics import YOLO
import cv2
import cvzone
import math

# cap = cv2.VideoCapture(0) # for webcam
cap = cv2.VideoCapture('/home/anubhav/pyprojects/us/input_video/videoplayback.mp4') # for video
# cap.set(3,1200)
# cap.set(4,700)


model = YOLO('../yolo_weights/yolov8n.pt')
model2 = YOLO('../yolo_weights/yolov8n-face.pt')

classnames = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "sink", "bathtub", "refrigerator", "microwave", "oven", "toaster", "sink", "remote control", "keyboard", "mouse", "laptop computer", "computer monitor", "cell phone", "potted plant", "book", "clock", "vase", "scissors", "teddy bear", "hair dryer", "toothbrush"]#

person_count = 10
while True:
  success, img = cap.read()
  results = model(img,stream=True)
  for r in results:
    boxes = r.boxes # to get bounding box of each of the result
    for box in boxes: # to loop through boxes
      # bounding box
      x1, y1, x2, y2 = box.xyxy[0] # only for first element
      x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
      
      # confidence
      conf = math.ceil((box.conf[0]*100))/100
      # classname
      clss = int(box.cls[0])
      currentclass = classnames[clss]
      if currentclass == "person" and conf>0.89:
        print(conf)
        person_count+=1
        src = f'/home/anubhav/pyprojects/us/body/{person_count}.jpg'
        cv2.imwrite(src, img[y1:y2, x1:x2])
        print('body saved')
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,200,0),3)
        cv2.putText(img,f'{conf} {classnames[clss]}',(x1,y1),cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 0), 2)
      elif currentclass == "person":
        print(conf)
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,200,0),3)
        cv2.putText(img,f'{conf} {classnames[clss]}',(x1,y1),cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 0), 2)
  
  cv2.imshow('Image',img)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break