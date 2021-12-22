import colorsys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import cv2


import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import image_preporcess
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
def add_bounding_box(out_boxes,out_classes,class_names, image):
    #print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
    text_size=3
    thickness = (image.shape[0] + image.shape[1]) // 600
    fontScale=1
    ObjectsList = []
     # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / len(class_names), 1., 1.)
                    for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),colors))
    
    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        

        label = '{}'.format(predicted_class)
        #label = '{}'.format(predicted_class)
       

        left, top, right, bottom = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.shape[0], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.shape[1], np.floor(right + 0.5).astype('int32'))

        mid_h = (bottom-top)/2+top
        mid_v = (right-left)/2+left

        # put object rectangle
        cv2.rectangle(image, (left, top), (right, bottom), colors[c], thickness)

        # get text size
        (test_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, thickness/text_size, 1)

        # put text rectangle
        cv2.rectangle(image, (left, top), (left + test_width, top - text_height - baseline), colors[c], thickness=cv2.FILLED)

        # put text above rectangle
        cv2.putText(image, label, (left, top-2), cv2.FONT_HERSHEY_SIMPLEX, thickness/text_size, (0, 0, 0), 1)

        # add everything to list
        ObjectsList.append([top, left, bottom, right, mid_v, mid_h, label])

    return image, ObjectsList
def visualize(out_boxes,out_classes,class_names,image_path):

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image_color = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        r_image, ObjectsList = add_bounding_box(out_boxes,out_classes,class_names,original_image_color)
        return r_image, ObjectsList



annotation_path="/Volumes/MLData/Python/Breast_Detector/annotations.txt"
image_directory="/Volumes/MLData/Python/Breast_Detector/yolo_extract/"
classes_path = "/Volumes/MLData/Python/Breast_Detector/classes.txt"
with open(classes_path) as f:
    class_names = f.readlines()
class_names = [c.strip() for c in class_names]
with open(annotation_path) as fp:
    for i, line in enumerate(fp):
        if i == 15:
            splits=line.split(" ")
            file_name=splits[0]
            annotations=splits[1:]
            boxes=[]
            classes=[]
            for ann in annotations:
                ann_splits=ann.split(",")
                boxes.append([int(ann_splits[0]),int(ann_splits[1]),int(ann_splits[2]),int(ann_splits[3])])
                classes.append(int(ann_splits[4]))
            print(boxes)
            r_image, ObjectsList=visualize(boxes,classes,class_names,image_directory+file_name)
            cv2.imshow(file_name, r_image)
            
            if cv2.waitKey(0) :
                cv2.destroyAllWindows()
        elif i > 200:
            break