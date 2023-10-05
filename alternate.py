from ultralytics import YOLO
import cv2
import cvzone
import math

# cap = cv2.VideoCapture(0) # for webcam
cap = cv2.VideoCapture('/home/anubhav/pyprojects/us/img/videoplayback.mp4') # for video
# cap.set(3,1200)
# cap.set(4,700)


model = YOLO('../yolo_weights/yolov8n.pt')
model2 = YOLO('../yolo_weights/yolov8n-face.pt')

classnames = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "sink", "bathtub", "refrigerator", "microwave", "oven", "toaster", "sink", "remote control", "keyboard", "mouse", "laptop computer", "computer monitor", "cell phone", "potted plant", "book", "clock", "vase", "scissors", "teddy bear", "hair dryer", "toothbrush"]#

person_count = 0
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
        classnames2 = ["face"]
        sets = model2(img,stream=True)
        for s in sets:
          boxes2 = s.boxes
          for box2 in boxes2:
            # bounding box
            x12, y12, x22, y22 = box2.xyxy[0] # only for first element
            x12, y12, x22, y22 = int(x12), int(y12), int(x22), int(y22)
            # confidence
            conf2 = math.ceil((box2.conf[0]*100))/100
            # classname
            clss2 = int(box2.cls[0])
            currentclass2 = classnames2[clss2]
            if conf2>0.70:
              print('face_captured')
              person_count+=1
              src = f'/home/anubhav/pyprojects/us/body/face_{person_count}.jpg'
              cv2.imwrite(src, img[y12:y22, x12:x22])
              cv2.rectangle(img,(x12,y12),(x22,y22),(0,200,0),3)
              cv2.putText(img,f'{conf2} {classnames2[clss2]}',(x12,y12),cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 0), 2)
            else:
              print('face not captured')
              cv2.rectangle(img,(x12,y12),(x22,y22),(0,200,0),3)
              cv2.putText(img,f'{conf2} {classnames2[clss2]}',(x12,y12),cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 0), 2)


        cv2.rectangle(img,(x1,y1),(x2,y2),(0,200,0),3)
        cv2.putText(img,f'{conf} {classnames[clss]}',(x1,y1),cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 0), 2)
      elif currentclass == "person":
        print(conf)
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,200,0),3)
        cv2.putText(img,f'{conf} {classnames[clss]}',(x1,y1),cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 0), 2)
  
  cv2.imshow('Image',img)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break