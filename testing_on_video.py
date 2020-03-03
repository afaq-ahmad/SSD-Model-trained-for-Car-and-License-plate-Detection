import cv2
from vsurv_parser import vsurvparser
import glob
import numpy as np
import sys

Video_path=sys.argv[1]
video_save_path=sys.argv[2]

model_path='vsurv_model.pb'
vsurvparse=vsurvparser(model_path,threshold_confidence=0.5)
#PATH_TO_IMAGE='image2.png'

vidcap = cv2.VideoCapture(Video_path)

size = (int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')#(*'MP4V')
out = cv2.VideoWriter(video_save_path,fourcc, 30, size)

try:
    while True:
        success,image = vidcap.read()
        if image is None:
            break
        All_Coordinates=vsurvparse.car_plate_from_image(image)
        for i in range(len(All_Coordinates)):
            text,xmin,ymin,xmax,ymax=All_Coordinates[i]
            text,xmin,ymin,xmax,ymax
            if text==1:
                text='Car'
                image=cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (255,51,51), 4)
                image=cv2.putText(image, text, (xmin,ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,51,51), 4)
            if text==2:
                text='License Plate'
                image=cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (51,51,255), 4)
                image=cv2.putText(image, text, (xmin,ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 4)
        out.write(image)
    out.release()
    vidcap1.release()
    vidcap2.release()

except KeyboardInterrupt:
    out.release()
    vidcap1.release()
    vidcap2.release()