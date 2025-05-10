# Tennis YOLO

This project analyzes Tennis players in a video to measure their speed, ball shot speed and number of shots. A YOLO model is used for tracking player and another for tracking the tennis ball by finetuning it. Further, a pretrained CNN (ResNet 50) is finetuned for detecting keypoints on the court. 

The dataset used for tennis ball detection sourced from https://universe.roboflow.com/viren-dhanwani/tennis-ball-detection/dataset/6
The dataset used for court keypoint extraction sourced from: https://github.com/yastrebksv/TennisCourtDetector?tab=readme-ov-file