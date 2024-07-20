from http.client import IncompleteRead
import cv2
import urllib.request
import numpy as np
import time

# Replace the URL with the IP camera's stream URL
url = 'http://192.168.0.115/cam-mid.jpg'
print('Opening Window')
cv2.namedWindow("live Cam Testing", cv2.WINDOW_AUTOSIZE)


time.sleep(5)
print('Displaying...')

# Read and display video frames
while True:
    # Read a frame from the video stream
    time.sleep(5)
    try: 
        print('Requesting..')
        img_resp=urllib.request.urlopen(url)
        print(img_resp)
        imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
        #ret, frame = cap.read()
        im = cv2.imdecode(imgnp,-1)

        cv2.imshow('live Cam Testing',im)
    except IncompleteRead:
        continue
    key=cv2.waitKey(5)
    if key==ord('q'):
        break
    

# cap.release()
cv2.destroyAllWindows()