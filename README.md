# yolov4-deepsort

The Google Drive folder containing our deliverables can be found [here](https://drive.google.com/drive/folders/15Pl7nEp11lsSThYhbWsnSvAdteqKZAdh?usp=sharing).


__TO TA__

- We uploaded the whole repo as the detector.py has dependencies all over it. Milestone 1 can be found in milestone1.ipynb or milestone1.py, milestone 2 in milestone2.py, and milestone 3 in the detector.py file. The report is found as report.pdf. Please, let us know if you have any questions!
- The notebooks for the milestones use OpenCV for webcam capturing. So you should run them locally, not in Colab.
- To run milestone 1, create a new conda environment in the terminal: `conda create -n deepsort python=3.9.13`. Then navigate to the this folder in the terminal, and run `pip3 install -r requirements1.txt`, followed by `python3 milestone1.py` to run the detector on your webcam.
- To run milestone 2, create a new conda environment in the terminal: `conda create -n deepsort python=3.8.13`. Then navigate to this folder in the terminal, and run `pip3 install -r requirements2.txt`, followed by `python milestone2.py -video 0` to test on a webcam. Note: Tensorflow does not work on the M1 chip, if you still want to run the code see [this link](https://stackoverflow.com/questions/65383338/zsh-illegal-hardware-instruction-python-when-installing-tensorflow-on-macbook.)
- for running detector.py, use `requirements-gpu.txt` as this is customized for the V100 on Loomo

Below, you see two example images from our milestone 1 initialization with pose estimation:

![Alt text](/images/1.jpg?raw=true "Title")
![Alt text](/images/2.jpg?raw=true "Title")


This repo is forked from: https://github.com/theAIGuysCode/yolov4-deepsort to customize for an own application. We use the DeepSort with Yolo algorithm to create a tracker which is partcipating in the Tandem Race at EPFL as part of the course CIVIL-459: Deep Learning for autonomous vehicles. The relevant additions can be found in the file detector.py, which is the file that runs the tracker on the Loomo. 

### References  

   Huge shoutout goes to hunglc007 and nwojke for creating the backbones of this repository:
  * [tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite)
  * [Deep SORT Repository](https://github.com/nwojke/deep_sort)
  * https://github.com/theAIGuysCode/yolov4-deepsort
