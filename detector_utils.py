import cv2
import os
import tensorflow as tf
from mrcnn.config import Config
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def noise_reduction(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(gray, (21, 21), 0)
    return image

def morph_ops(image, kernel):
    image = cv2.threshold(image,25,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    image = cv2.dilate(image, kernel, iterations=1)
    return image
    
def draw_boxes(image, cnts):
    for c in cnts:
        if cv2.contourArea(c) <500:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image


class TestConfig(Config):
     NAME = "test"
     GPU_COUNT = 1
     IMAGES_PER_GPU = 1
     NUM_CLASSES = 1 + 80

class load_dl():
    def __init__(self):
        from mrcnn.model import MaskRCNN
        from keras.preprocessing.image import img_to_array
        self.MaskRCNN = MaskRCNN
        self.img_to_array = img_to_array
        
    def model(self):
        rcnn = self.MaskRCNN(mode='inference', model_dir='./', config=TestConfig())
        rcnn.load_weights("mask_rcnn_coco.h5", by_name=True)
        return rcnn
    
    def to_array(self,img):
        return self.img_to_array(img)

def load_ip():
    return cv2.absdiff

def load_ml():
    return cv2.createBackgroundSubtractorKNN()

