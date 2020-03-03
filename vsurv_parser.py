import os,sys,glob,time
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import camelot
import tabula
import fitz

#from Table_structure_extraction import extract_table
def non_max_suppression_fast(boxes, overlapThresh=0.9):
	if len(boxes) == 0:
		return []
 
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
 
	# initialize the list of picked indexes	
	pick = []
 
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
 
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
 
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
 
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
 
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
 
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
 
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")


model_path='model/ssd_model.pb'

class vsurvparser:
    """
    Class to detect and extract tables and its information from an image.
    Parameters:
        model_path: Tensorflow Model Path
        threshold_confidence: (optional) Detected tables threshold confidence value. Default set at 0.5.
    """
    
    def __init__(self, model_path,threshold_confidence=0.5):    
        self.model_path=model_path
        self.threshold_confidence=threshold_confidence
        
        # Grab path to current working directory
        CWD_PATH = os.getcwd()
        # Path to frozen detection graph .pb file, which contains the model that is used
        # for object detection.
        PATH_TO_CKPT = os.path.join(CWD_PATH,model_path)

        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        # Load the Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            sess = tf.Session(graph=detection_graph)

        # Input tensor is the image
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        # Number of objects detected
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        
        self.detection_boxes=detection_boxes
        self.detection_scores=detection_scores
        self.detection_classes=detection_classes
        self.num_detections=num_detections
        self.image_tensor=image_tensor
        self.sess=sess
    def car_plate_from_image(self,image):
        
        """
        Function to detect and extract tables and its information from an image.
        
        Input Parameters:
        ----------------
            image: Numpy array of image in BGR(open-cv) format.
        
        Returns:
        --------
        All_Coordinates,All_DataFrames:      list,list
        
        
        
        """
        
        image_expanded = np.expand_dims(image, axis=0)
        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_expanded})
        
        hight,width,channel=image.shape
        indexes_f=np.where(scores[0]>self.threshold_confidence)[0]
        #-----------------------------------
        
        coordinates_initial=[]
        for p in range(len(indexes_f)):
            y1,x1,y2,x2=boxes[0][[indexes_f[p]]][0]
            class_predicted=classes[0][p]
            x1,y1,x2,y2=width*x1,hight*y1,width*x2,hight*y2
            coordinates_initial.append([class_predicted,int(x1),int(y1),int(x2),int(y2)])
            
        coordinates_initial=np.array(coordinates_initial)
        

        return coordinates_initial.astype(int)