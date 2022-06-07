# yolov4-deepsort

This repo is forked from: https://github.com/theAIGuysCode/yolov4-deepsort to customize for an own application. We use the DeepSort with Yolo algorithm to create a tracker which is partcipating in the Tandem Race at EPFL as part of the course CIVIL-459: Deep Learning for autonomous vehicles. The relevant additions can be found in the file detector.py, which is the file that runs the tracker on the Loomo. 

Object tracking implemented with YOLOv4, DeepSort, and TensorFlow. YOLOv4 is a state of the art algorithm that uses deep convolutional neural networks to perform object detections. We can take the output of YOLOv4 feed these object detections into Deep SORT (Simple Online and Realtime Tracking with a Deep Association Metric) in order to create a highly accurate object tracker.

### References  

   Huge shoutout goes to hunglc007 and nwojke for creating the backbones of this repository:
  * [tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite)
  * [Deep SORT Repository](https://github.com/nwojke/deep_sort)
  * https://github.com/theAIGuysCode/yolov4-deepsort
