# Detector
This Repository consists of three methods for object detecion.

    1) Background subtraction using absolute difference.
    
    2) Background subtraction using KNN.
    
    3) Pre-trained MaskRCNN on coco dataset

# Installation Steps
`pip install -r requirements.txt`

Download pre-trained weights from `https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5`

Run `python main.py`

Select method by giving input as ML, IP, DL `input =  "ML"`