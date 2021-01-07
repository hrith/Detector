import cv2
import numpy as np

class Utils():
    def noise_reduction(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.GaussianBlur(image, (21, 21), 0)
        return image

    def morph_ops(self, image, kernel, tp):
        image = cv2.threshold(image,25,255,tp)[1]
        image = cv2.dilate(image, kernel, iterations=1)
        return image
    
    def draw_boxes(self, image, cnts):
        for c in cnts:
            if cv2.contourArea(c) <500:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return image

    def draw_rois(self, image, cnts, cls):
        CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        for i in range(len(cnts)):
            (y1, x1, y2, x2) = cnts[i]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            image = cv2.putText(image, CLASS_NAMES[cls[i]], (x1-10,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA) 
        return image
        
    
class Detector(Utils):
    def __init__(self, algo):
        from detector_utils import load_ip, load_ml, load_dl
        self.algo = algo
        cmds = {"IP":[load_ip,
                      self.ip_detector],
                "ML":[load_ml,
                    self.ml_detector],
                "DL":[load_dl,
                      self.dl_detector]}[algo]

        
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened:
            assert 'Unable to open: ' + args.input
        else:
            
            self.model = cmds[0]()
            self.backSub = cv2.createBackgroundSubtractorKNN()
            self._start(cmds[1])
            self.capture.release()
            cv2.destroyAllWindows()

    def ml_detector(self, frame):
        op = self.noise_reduction(frame)
        op = self.model.apply(op)
        op = self.morph_ops(op, np.ones((5,5),np.uint8), cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(op, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return cnts

    def ip_detector(self, frame):
        op = self.noise_reduction(frame)
        try:
            self.firstFrame
        except:
            self.firstFrame = op
            return []
        op = self.model(self.firstFrame, op)
        op = self.morph_ops(op, np.ones((5,5),np.uint8), cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(op, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return cnts

    def dl_detector(self, frame):
        results = self.model.model().detect([self.model.to_array(frame)], verbose=0)
        return [results[0]['rois'], results[0]['class_ids']]
        
    def _start(self, ex_func):
        while True:
            ret, frame = self.capture.read()
            if frame is None:
                break
            coords = ex_func(frame)
            if len(coords) > 0:
                if self.algo != "DL":
                    frame = self.draw_boxes(frame, coords)
                else:
                    frame = self.draw_rois(frame, coords[0], coords[1])
            cv2.imshow('Frame', frame)
    
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
              break
        

if __name__ == "__main__":
    algo = input()
    if algo in ["IP", "ML", "DL"]:
        Detector(algo)
