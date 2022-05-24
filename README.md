# yolov4-deepsort


This repo is forked from: https://github.com/theAIGuysCode/yolov4-deepsort to customize for an own application. All credit for the deep sort code goes to: https://github.com/theAIGuysCode. 

Run: final_home_test.ipynb (for our usage changed client.py; outputs a red box in the middle of the frame if the triggered person is not shown & outputs a green box in the center of the bounding box if triggered person is in frame)

Changes from now on please in detector.py  

Object tracking implemented with YOLOv4, DeepSort, and TensorFlow. YOLOv4 is a state of the art algorithm that uses deep convolutional neural networks to perform object detections. We can take the output of YOLOv4 feed these object detections into Deep SORT (Simple Online and Realtime Tracking with a Deep Association Metric) in order to create a highly accurate object tracker.

### References  

   Huge shoutout goes to hunglc007 and nwojke for creating the backbones of this repository:
  * [tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite)
  * [Deep SORT Repository](https://github.com/nwojke/deep_sort)
