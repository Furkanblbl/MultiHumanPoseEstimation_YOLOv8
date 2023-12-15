# Multi_Human_Pose_Estimation_YOLOv8
 By detecting the event, an improvement can be made in such a way that how long he lay down on the ground, how long he squatted, how long he ran, and how long he was detected in an action other than these 3 actions, and followed up, will be immediately visible to many people.

## Setup 
pip install git+https://github.com/ultralytics/yolov5.git
pip install opencv-python
pip install numpy



## Example Usage

#### Initialization:
```python
# Create a VideoProcessor object by providing the YOLO model path and the video source
model_path = "yolov8m-pose.pt"
video_source = "videos/media4.mp4"
video_processor = VideoProcessor(model_path, video_source)
```
#### Processing Video:
```python
# Process the video to analyze human movements
video_processor.process_video()
```
### Expected Output:
- The script will read the specified video file frame by frame.
- For each frame, it will detect humans using the YOLO model.
- It will analyze the movement status of each individual detected in the frame.
- The script will display on the screen the type of action detected for each person and the duration of that action.

### Usage Instructions:
1. Ensure you have the required libraries installed (`ultralytics`, `cv2`, `numpy`, `time`).
2. Replace `yolov8m-pose.pt` with the correct YOLO model path if different.
3. Adjust the `video_source` variable to the location of your desired video file.
4. Run the script in a Python environment.
5. Observe the output on the screen, displaying the detected actions and their durations for each person throughout the video.


## Examples

# Create a VideoProcessor object by providing the YOLO model path and the video source
model_path = "yolov8n-pose.pt"
video_source = "videos/media4.mp4"
video_processor = VideoProcessor(model_path, video_source)

# Process the video to analyze human movements
video_processor.process_video()

# Expected Output:
The script will read the specified video file frame by frame.
For each frame, it will detect humans using the YOLO model.
It will analyze the movement status of each individual detected in the frame.
The script will display on the screen the type of action detected for each person and the duration of that action.

# Usage Instructions:
Ensure you have the required libraries installed (ultralytics, cv2, numpy, time).
Replace yolov8m-pose.pt with the correct YOLO model path if different.
Adjust the video_source variable to the location of your desired video file.
Run the script in a Python environment.
Observe the output on the screen, displaying the detected actions and their durations for each person throughout the video.