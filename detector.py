import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import core.utils as utils
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

class Detector(object):
    def __init__(self):        
        # definition of the parameters
        self.nms_max_overlap = 1.0
        self.counter_reinit = 0
        self.frame_num = 0
        self.first_pass = True
        self.input_size = 416
        self.trigger_id = None

        # initialize deep sort
        model_filename = 'model_data/mars-small128.pb'
        self.encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        
        # calculate cosine distance metric
        max_cosine_distance = 0.3
        nn_budget = None #Fix samples per class to at most this number. Removes the oldest samples when the budget is reached.
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        
        # initialize tracker
        max_iou_distance = 0.45
        max_age = 60
        n_init = 10
        self.tracker = Tracker(metric, max_iou_distance, max_age, n_init)

        # load saved model
        self.saved_model_loaded = tf.saved_model.load('./checkpoints/yolov4-416', tags=[tag_constants.SERVING])
        self.infer = self.saved_model_loaded.signatures['serving_default']

    def forward(self, frame):
        self.frame_num +=1

        # prepare image for detection process
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_data = cv2.resize(frame, (self.input_size, self.input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        # divide into batches
        batch_data = tf.constant(image_data)
        pred_bbox = self.infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        # predict bounding boxes of all objects in frame
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=0.45,
            score_threshold=0.50
        )
        #print(valid_detections)

        # reinitialization when camera is blocked for several seconds
        if valid_detections[0] == 0:
            self.counter_reinit += 1
            if self.counter_reinit >= 61:
                print('reinitialized')
                self.first_pass = True
                self.counter_reinit = 0

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config 
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)
        allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only humans to be tracked
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)

        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = self.encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # call the tracker
        self.tracker.predict()
        self.tracker.update(detections)
        self.in_frame = False

        # update tracks
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 

            bbox = track.to_tlbr()
            class_name = track.get_class()

            # initialize triggered person
            if self.first_pass == True:
                self.trigger_id = track.track_id
                self.first_pass = False

            # show only triggered person in frame
            if track.track_id == self.trigger_id:
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,0,0), 2)
                self.in_frame = True
                
        if self.in_frame:
            return [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2], [1]
        else:
            return [frame.shape[1]/2, frame.shape[0]/2], [0] #center of frame