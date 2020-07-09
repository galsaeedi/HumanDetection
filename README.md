# HumanDetection

## Introduction
I've worked on the problem of human detection,face detection, eye detection. This detector is capable of detecting a human and his body parts in an image Uinsg TensorFlow's Object Detection API. Currently, it detects just the human body, face, eye, arm and legs. A new version can be developed to detect more parts and objects. <br><br>This code is written on cloud,but the general procedure can also be used for Linux or any operating systems, just file paths, package installation and environment settings will need to change accordingly.

## Requirements
<ul>
<li><b>Python3</b></li> <br>
<li><b>Tensorflow 1.15</b>: Download the full TensorFlow object detection repository located at https://github.com/tensorflow/models</li> <br>
<li><b>pre-trained classifiers</b>: TensorFlow provides several object detection models (pre-trained classifiers with specific neural network architectures) in its <a href="https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md">model zoo</a>. I re-trained my detector on the Faster-RCNN-Inception-V2 model, the detection worked very well, but with a noticeably slow speed. </li>
<li><b>Opencv</b> [v3]</li> <br>
<li><b>python libraries</b>: Here is a list of all the python dependencies
  <ul><li>tensorflow-gpu 1.15</li>
  <li>opencv-python</li>
  <li>pandas</li>
  <li>numpy</li>
  </ul>
  </ll>

</ul>
<b>note:</b> This code runs on the cloud. If you work locally, it is recommended to install Anaconda, CUDA and cuDNN. <a href="https://www.tensorflow.org/install/source#tested_build_configurations"> Here </a> is a table showing which version of TensorFlow requires which versions of CUDA and cuDNN.

## Installation
<ul>
  
  <li>After gathering or downloading the dataset. First, split the data into train and test. Then, you need to run the python txt_to_csv.py file, to convert the txt labels     files to one csv file 
 <br><code>
  python txt_to_csv.py
 </code><br>
  This creates a train_labels.csv and test_labels.csv file in the \object_detection\images folder.
  <br><b>Download data processing scripts from <a href="https://github.com/galsaeedi/OIDv4_ToolKit"> my repo  </a></b>
  </li>
  
  <br><li>Run the python generate_tfrecord.py generate the TFRecords that serve as input data to the TensorFlow training model.
  <br><code>
  python generate_tfrecord.py
  </code><br>
  </li>
  
  <br><li>To start train run
  <br><code>
  python train.py
  </code><br>
    </li>
  
  <br><li>
  View the progress of the training job by using TensorBoard.
  <br><code>
  %load_ext tensorboard <br>
  %tensorboard --logdir training 
  </code>
  </li>
  </ul>
  
## Performance of code
<img src="">

## Results

## To do 

## Thanks to:
