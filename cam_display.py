import cv2
import matplotlib.pyplot as plt
import urllib.request
import numpy as np
import concurrent.futures
import time
from ultralytics import YOLO
from gtts import gTTS
from playsound import playsound
import os
import tempfile
import streamlit as st
import numpy as np
import pandas as pd
from collections import Counter


from http.client import IncompleteRead
 
url='http://192.168.0.102/cam-hi.jpg'
im=None

# Translation dictionary
translations = {
    'person': 'orang',
    'bicycle': 'sepeda',
    'car': 'mobil',
    'motorcycle': 'motor',
    'airplane': 'pesawat',
    'bus': 'bus',
    'train': 'kereta api',
    'truck': 'truk',
    'boat': 'perahu',
    'traffic light': 'lampu lalu lintas',
    'fire hydrant': 'hydrant',
    'stop sign': 'rambu berhenti',
    'parking meter': 'meteran parkir',
    'bench': 'bangku',
    'bird': 'burung',
    'cat': 'kucing',
    'dog': 'anjing',
    'horse': 'kuda',
    'sheep': 'domba',
    'cow': 'sapi',
    'elephant': 'gajah',
    'bear': 'beruang',
    'zebra': 'zebra',
    'giraffe': 'jerapah',
    'backpack': 'ransel',
    'umbrella': 'payung',
    'handbag': 'tas tangan',
    'tie': 'dasi',
    'suitcase': 'koper',
    'frisbee': 'frisbee',
    'skis': 'ski',
    'snowboard': 'papan seluncur salju',
    'sports ball': 'bola olahraga',
    'kite': 'layang-layang',
    'baseball bat': 'tongkat baseball',
    'baseball glove': 'sarung tangan baseball',
    'skateboard': 'papan luncur',
    'surfboard': 'papan selancar',
    'tennis racket': 'raket tenis',
    'bottle': 'botol',
    'wine glass': 'gelas anggur',
    'cup': 'cangkir',
    'fork': 'garpu',
    'knife': 'pisau',
    'spoon': 'sendok',
    'bowl': 'mangkuk',
    'banana': 'pisang',
    'apple': 'apel',
    'sandwich': 'roti lapis',
    'orange': 'jeruk',
    'broccoli': 'brokoli',
    'carrot': 'wortel',
    'hot dog': 'hot dog',
    'pizza': 'pizza',
    'donut': 'donat',
    'cake': 'kue',
    'chair': 'kursi',
    'couch': 'sofa',
    'potted plant': 'tanaman dalam pot',
    'bed': 'tempat tidur',
    'dining table': 'meja makan',
    'toilet': 'toilet',
    'tv': 'televisi',
    'laptop': 'laptop',
    'mouse': 'mouse',
    'remote': 'remote',
    'keyboard': 'keyboard',
    'cell phone': 'ponsel',
    'microwave': 'microwave',
    'oven': 'oven',
    'toaster': 'pemanggang roti',
    'sink': 'wastafel',
    'refrigerator': 'kulkas',
    'book': 'buku',
    'clock': 'jam',
    'vase': 'vas',
    'scissors': 'gunting',
    'teddy bear': 'boneka beruang',
    'hair drier': 'pengering rambut',
    'toothbrush': 'sikat gigi'
}


# Function to translate detected objects
def translate_labels(labels):
    return [[translations[label], position] if label in translations else label for label, position in labels]

def determine_position(x1, x2, image_width):
    # Calculate the midpoint of the bounding box
    midpoint = (x1 + x2) / 2
    # Define thresholds for "left", "middle", "right"
    if midpoint < image_width / 4:
        return "kiri"
    elif midpoint < 3 * image_width / 4:
        return "depan"
    else:
        return "kanan"

def run_app():
    st.title("Live Object Detection")

    # Placeholder for live video feed
    frame_placeholder = st.empty()
    
    # Placeholder for pie chart
    chart_placeholder = st.empty()

    model = YOLO('yolov8n.pt')  # Load YOLOv8 Nano model
    detected_counts = Counter()

    while True:
        time.sleep(0.1)
        try:
            img_resp = urllib.request.urlopen(url)
            imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            im = cv2.imdecode(imgnp, -1)

            image_width = im.shape[1]  # Get the width of the image

            results = model(im)  # Detect objects
            detected_objects = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    label = model.names[int(box.cls[0])]


                    position = determine_position(x1, x2, image_width)

                    detected_objects.append([label, position])

                    cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(im, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if len(detected_objects) > 0:
                detected_objects = translate_labels(detected_objects)
                detected_counts.update([objects for objects, position in detected_objects])

                object_text = [label + " di " + position for label, position in detected_objects]
                object_text = list(set(object_text))
                text = ",".join(object_text)
                # TTS code is removed for brevity, can be included as needed

                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio:
                    temp_audio_path = temp_audio.name
                    tts = gTTS(text=text, lang='id')
                    tts.save(temp_audio_path)

                try:
                    playsound(temp_audio_path)
                    time.sleep(0.5 + 0.5 * len(object_text))
                except Exception as e:
                    print(f"Error playing sound: {e}")

                try:
                    os.remove(temp_audio_path)
                except Exception as e:
                    print(f"Error deleting sound: {e}")


            frame_placeholder.image(im, channels="BGR", use_column_width=True)
            data = pd.DataFrame.from_dict(detected_counts, orient='index', columns=['count'])
            chart_placeholder.bar_chart(data)

        except Exception as e:
            print(e)
            continue


    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_app()