        

#!/usr/bin/env python3

"""
Tiny-yolo-v4 device side decoding demo
The code is the same as for Tiny-yolo-V3, the only difference is the blob file.
The blob was compiled following this tutorial: https://github.com/TNTWEN/OpenVINO-YOLOV4
"""

from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time



import os 
import imutils
import subprocess
from gtts import gTTS 
from pydub import AudioSegment

# Get argument first
nnPath = 'tiny-yolo-v4_openvino_2021.2_6shave.blob'

# tiny yolo v4 label texts
labelMap = [
    "person",         "bicycle",    "car",           "motorbike",     "aeroplane",   "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",   "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",       "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",    "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball", "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",      "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",      "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",       "donut",         "cake",
    "chair",          "sofa",       "pottedplant",   "bed",           "diningtable", "toilet",        "tvmonitor",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",  "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",       "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"
]

syncNN = True

# Create pipeline
pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_2)

# Define sources and outputs
camRgb = pipeline.createColorCamera()
detectionNetwork = pipeline.createYoloDetectionNetwork()
xoutRgb = pipeline.createXLinkOut()
nnOut = pipeline.createXLinkOut()
xoutRgb1 = pipeline.createXLinkOut()

xoutRgb.setStreamName("rgb")
nnOut.setStreamName("nn")
xoutRgb1.setStreamName("video")

# # Properties
camRgb.setPreviewSize(416, 416)
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)         #1080
# camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)      #3040
# camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)          #2160
camRgb.setVideoSize(1000, 700) ## CUSTOMIZE SIZE
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)


xoutRgb1.input.setBlocking(False)
xoutRgb1.input.setQueueSize(1)
camRgb.video.link(xoutRgb1.input)



# Network specific settings
detectionNetwork.setConfidenceThreshold(0.5)
detectionNetwork.setNumClasses(80)
detectionNetwork.setCoordinateSize(4)
detectionNetwork.setAnchors(np.array([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319]))
detectionNetwork.setAnchorMasks({"side26": np.array([1, 2, 3]), "side13": np.array([3, 4, 5])})
detectionNetwork.setIouThreshold(0.5)
detectionNetwork.setBlobPath(nnPath)
detectionNetwork.setNumInferenceThreads(2)
detectionNetwork.input.setBlocking(False)

# Linking
camRgb.preview.link(detectionNetwork.input)
if syncNN:
    detectionNetwork.passthrough.link(xoutRgb.input)
else:
    camRgb.preview.link(xoutRgb.input)

detectionNetwork.out.link(nnOut.input)

texts=[]
detected=[]
# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
    out = device.getOutputQueue(name="video", maxSize=1, blocking=False)

    frame = None
    detections = []
    startTime = time.monotonic()
    counter = 0
    color2 = (255, 255, 255)


    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def displayFrame(name, frame):
        color = (0, 0, 255)
        height = frame.shape[0]
        width  = frame.shape[1]
        for detection in detections:
            x1 = int(detection.xmin * width)
            x2 = int(detection.xmax * width)
            y1 = int(detection.ymin * height)
            y2 = int(detection.ymax * height)
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
            cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            label =labelMap[detection.label]
            # print(label)
            if k == ord('d'):
                centerX, centerY = x1,y1            
                if centerX <= width/3:
                    W_pos = "left "
                elif centerX <= (width/3 * 2):
                    W_pos = "center "
                else:
                    W_pos = "right "
                
                if centerY <= height/3:
                    H_pos = "top "
                elif centerY <= (height/3 * 2):
                    H_pos = "mid "
                else:
                    H_pos = "bottom "
                # texts.append("obstical detected at "+H_pos + W_pos + "as" +label)
                # texts.append(label +" detected at "+H_pos + W_pos ) 
                texts.append(label +"  at "+H_pos + W_pos )            
                detected.append(label)
                print(texts)            


                description = ', '.join(texts)
                tts = gTTS(description, lang='en')
                tts.save('tts.mp3')
                tts = AudioSegment.from_mp3("tts.mp3")
                subprocess.call(["ffplay", "-nodisp", "-autoexit", "tts.mp3"])
            else:
                pass

        # Show the frame
        cv2.imshow(name, frame)

    while True:
        k=0xFF & cv2.waitKey(1)
        if syncNN:
            inRgb = qRgb.get()
            inDet = qDet.get()
            output = out.get()  
        else:
            inRgb = qRgb.tryGet()
            inDet = qDet.tryGet()
            output = out.tryGet() 

        if inRgb is not None:
            frame = inRgb.getCvFrame()
            cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                        (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        if inDet is not None:
            detections = inDet.detections
            counter += 1

        if frame is not None:
            frame = output.getCvFrame()
            displayFrame("yolo", frame)

        if k == ord('q'):
            break


cv2.destroyAllWindows()
os.remove("tts.mp3")      
