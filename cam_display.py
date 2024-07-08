import cv2
import matplotlib.pyplot as plt
import cvlib as cv
import urllib.request
import numpy as np
from cvlib.object_detection import draw_bbox
import concurrent.futures
import time
from ultralytics import YOLO

from http.client import IncompleteRead
 
url='http://192.168.0.104/cam-hi.jpg'
im=None
 
def run1():
    cv2.namedWindow("live transmission", cv2.WINDOW_AUTOSIZE)
    while True:
        time.sleep(0.1)
        try:
            img_resp=urllib.request.urlopen(url)
            imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
            im = cv2.imdecode(imgnp,-1)
    
            cv2.imshow('live transmission',im)
        except IncompleteRead:
            continue
        key=cv2.waitKey(5)
        if key==ord('q'):
            break
            
    cv2.destroyAllWindows()

def run2():
    model = YOLO('yolov8n.pt')  # Load YOLOv8 Nano model

    cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE)
    while True:
        time.sleep(0.1)
        try:
            print('detection - getting image')
            img_resp = urllib.request.urlopen(url)
            imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            im = cv2.imdecode(imgnp, -1)

            print('detection - detecting..')
            results = model(im)  # Detect objects

            print('detection - drawing..')
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    label = model.names[int(box.cls[0])]
                    cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(im, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            print('detection - showing..')
            cv2.imshow('detection', im)
        except Exception as e:
            print(e)
            continue

        key = cv2.waitKey(5)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

def run3():
    cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE)
    while True:
        time.sleep(0.1)
        # try:
        print('detection - getting image')
        img_resp=urllib.request.urlopen(url)
        imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
        im = cv2.imdecode(imgnp,-1)
        print('detection - detecting..')
        # time.sleep(2)

        bbox, label, conf = cv.detect_common_objects(im)
        # print('detection - drawing..')
        # im = draw_bbox(im, bbox, label, conf)

        print('detection - showing..')
        cv2.imshow('detection',im)
        # except Exception as e:
        #     print(e)
        #     continue
        key=cv2.waitKey(5)
        if key==ord('q'):
            break
            
    cv2.destroyAllWindows()
 
 
 
if __name__ == '__main__':
    print("started")
    run2()
    # with concurrent.futures.ProcessPoolExecutor() as executer:
    #         f1= executer.submit(run1)
    #         f2= executer.submit(run2)