import cv2
from vsurv_parser import vsurvparser
import glob
import numpy as np
import sys
image_folder=sys.argv[1]

Image_files=glob.glob(image_folder+'/*')
model_path='vsurv_model.pb'
vsurvparse=vsurvparser(model_path,threshold_confidence=0.5)
#PATH_TO_IMAGE='image2.png'



for PATH_TO_IMAGE in Image_files:
    image = cv2.imread(PATH_TO_IMAGE)
    if image is None:
        print ('No Image found at given path')
    else:
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
    cv2.imwrite('_predicted.'.join(PATH_TO_IMAGE.split('.')),image)