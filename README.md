## Visual Odometry: Vision Based Estimation of Motion for Autonomous Driving

### Abstract
Leveraging computer vision techniques, a system that processes sequential stereo camera images to estimate the motion of a moving vehicle, was designed. Through feature extraction, matching, and robust estimation algorithms, the system successfully tracked key points in consecutive frames, allowing for the precise calculation of the vehicle's pose and movement. The solution demonstrates the importance of vision for autonomous systems.

### Demo
(Buffering the GIF may take a few seconds)
![demo](https://github.com/TomGoesGitHub/Visual-Odometry/assets/81027049/8f9d312a-c75b-4a3c-a9fe-222c77248476)

### Dataset
The Datasat can be found here: https://www.cvlibs.net/datasets/kitti/eval_odometry.php

It contains stereo-vision image data (only greyscale images were used in this project), laser-scan data (not used in this project) and GPS ground truth data for 11 routes in total. All data was captured at a rate of 10 Hz. The sensor setup is shown below.

![image](https://github.com/TomGoesGitHub/Visual-Odometry/assets/81027049/d46b5348-5175-49cb-bd97-e37d2a43dd35)

### Problem Statement
In comparison to classical odomotry for wheeled vehicles, which uses velocity and/or acceleration data, visual odometry does not leverage those information. Instead, as the name suggests, visual odometry uses camera information only. 
Applications of Visual Odometry are of great importance for 1) unwheeled vehicles like drones or 2) when no GPS-signal is available for navigation as in indoor-, underwater- and space-applications.

The task is to solve the visual odometry problem, which attempts to determine the position and orientation of a robot by analyzing the associated camera frames. This project evaluates, how precisely the GPS-ground truth can be recovered from vision only.


### Implementation Details
A Feature-Based approach for stereo VO was used and implemented with OpenCV (Python). The architecture is shown below.
![Architecture](https://github.com/TomGoesGitHub/Visual-Odometry/assets/81027049/7d288e95-2623-4393-96fe-ecfcf243ebbd)


### Results
Experiments were carried out with a 2Hz frame-input rate.
![results_2Hz](https://github.com/TomGoesGitHub/Visual-Odometry/assets/81027049/1e2269bb-4951-4f43-86ae-06eab5bc132a)

