import cv2
import imutils
import numpy as np
greenLower = (135,135,135)
greenUpper = (255,255,255)
limit=0
limitx=[]
boxes=[]
Xboxes=[]
akey=0
classNames= []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def getObjects(img,thres,nms,draw=True,objects=[]):
    global akey
    global limit
    global limitx
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    #print(classIds,bbox)

    if len(objects) == 0: objects = classNames
    objectInfo =[]
    if len(classIds) != 0:

        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId - 1]
            if className in objects:
                akey=0
                objectInfo.append([box, className])

                boxes.append(box[1])
                Xboxes.append(box[0])

                for i in range(len(boxes)-2):
                    if i >0 :
                        if abs(boxes[i]-boxes[i-1])<15:
                            del boxes[i]
                            del Xboxes[i]
                        if (boxes[i]>boxes[i-1]) and (boxes[i]>boxes[i+1]) :
                            if boxes[i]<=limit :#and (Xboxes[i]<=max(limitx) or Xboxes[i]>=min(limitx)):
                                print(limitx)
                                cv2.circle(img, (Xboxes[i], boxes[i]), 10, color=(0, 255, 0), thickness=20)
                                cv2.putText(img,"in",(Xboxes[i], boxes[i]+10), cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),2)


                            #else:
                             #   print(limitx)
                              #  cv2.circle(img, (Xboxes[i], boxes[i]), 10, color=(255, 0, 0), thickness=20)
                               # cv2.putText(img,"out",(Xboxes[i], boxes[i]+10),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),2)




                print(boxes)
             #   if box[1] >200 :
             #   cv2.circle(img, (494,496), 2, color=(255, 0, 0), thickness=2)
                if (draw):
                    if box[0]>660 :

                        cv2.rectangle(img,box,color=(255,0,0),thickness=2)
                        cv2.putText(img,className.upper(),(box[0]+10,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),2)
                        cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),2)
                    if box[0]<620:
                        cv2.rectangle(img, box, color=(0, 0,255), thickness=2)
                        cv2.putText(img, className.upper(), (box[0] + 10, box[1] + 30),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                        cv2.putText(img, className.upper(), (box[0] + 10, box[1] + 30),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
                  #  if box[1] == 496:
                   #     print(box[0], box[1])
            else :
                akey+=1
                if (akey % 20) ==0:
                    akey=0
                    boxes.clear()
                    Xboxes.clear()





    return img,objectInfo

cap = cv2.VideoCapture('est.mp4')


cap.set(3, 1280)
cap.set(4, 720)
cap.set(5,20.0)
print (cap.get(cv2.CAP_PROP_FPS))

while True:
    success, img = cap.read()
    img = imutils.resize(img, width=1000,height=1000)
    blurred = cv2.GaussianBlur(img, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2BGR)

    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cv2.imshow("xxx",mask)
    #X=cv2.findNonZero(mask)
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)
    minLineLength = 100
    lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi / 180, threshold=100, lines=np.array([]),
                            minLineLength=minLineLength, maxLineGap=80)

    a, b, c = lines.shape

    for i in range(a):
        if(abs(lines[i][0][1]-lines[i][0][3])<7):
          #  cv2.line(img, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3,
           #          cv2.LINE_AA)
            limit=lines[i][0][1]
        if (abs(lines[i][0][0] - lines[i][0][2]) < 225):
         #   cv2.line(img, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3,
          #           cv2.LINE_AA)
            limitx.append(lines[i][0][0])



    ### converting to bounding boxes from polygon

    result, objectInfo = getObjects(img, 0.5, 0.5, objects=['sports ball'])
    cv2.namedWindow('Video Life2Coding', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Video Life2Coding', img)

    key=cv2.waitKey(1)
    if key == ord('p'):
        cv2.waitKey(-1)





