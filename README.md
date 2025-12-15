# VisiLoc
Visiloc is a camera-based indoor localization system which uses ArUco markers to estimate position and orientation in real-time, without the use of a GPS 

![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red)
![ROS2](https://img.shields.io/badge/ROS2-Humble-blue)
![Python](https://img.shields.io/badge/Python-3.10-green)

# How It Works:
Every 30 miliseconds the robot's camera captures an image and sends it over to the Visloc. The first thing Visiloc does is convert the image to grayscale and starts checking if any of them contain the specific bit pattern that matches our marker ID. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/76f7efe6-e7eb-4b84-be66-3887b47728bc" alt="Screenshot from 2025-12-14 22-54-31" height="350">
</p>
<p align="center"><em>Simulated Room</em></p>

Once the marker is located Opencv gives us two pieces of data: rvec (rotation vector) and tvec (translation vector), these vectors are used to used to describe the marker's 3D pose relative to the camera. Using these vectors we are able to find the location of the bot with respect to the marker postion in the world by using the formula ``` camera_position = -R^T × tvec ``` (where R is the rotation matrix).

<p align="center">
  <img src="https://github.com/user-attachments/assets/9358f16f-206f-4f58-a37d-aabbb05e941b" alt="Screenshot from 2025-12-14 22-54-31" height="350">
</p>
<p align="center"><em>Marker on North Wall</em></p>

Using some trigonometry and math, we can then rotate this camera position from the marker's coordinate frame into the world coordinate frame, giving us the robot's exact position in the world

| <img height="350" src="https://github.com/user-attachments/assets/7ff55763-387d-4e18-88d9-f59ec8ad3124"> | <img height="350" src="https://github.com/user-attachments/assets/a51c6f94-5cf1-4665-b965-344f519d6057"> | <img height="350" src="https://github.com/user-attachments/assets/f6317f64-d160-43a5-84b1-be96ad2b7d0f"> |
:-------------------------:|:-------------------------:|:-------------------------:
Simulated Bot | Anotated Camera Output | Calculated Location

---

| <img height="350" src="https://github.com/user-attachments/assets/468a994f-823e-4151-b702-f1cc61130cb3"> | <img height="350" src="https://github.com/user-attachments/assets/722f84f3-726a-42a2-a2f7-3f9f532d9b31"> | <img height="350" src="https://github.com/user-attachments/assets/3b1c5965-6d0e-4e92-adc7-533788457d56"> |
:-------------------------:|:-------------------------:|:-------------------------:
Simulated Bot | Anotated Camera Output | Calculated Location

# Try it for yourself:

1. clone the repo
```
git clone https://github.com/vedantmalkar/VisiLoc.git
cd VisiLoc/ros_ws/
```

2. build and source your workspace
```
colcon build
source install/setup.bash
```

3. launch simulation
```
ros2 launch ignition_robot complete.launch.py 
```

4. on seperate terminal run finder node
```
ros2 run vision_controller coordinate_finder 
```

5. on another terminal run visualizer node
```
ros2 run vision_controller coordinate_visualizer 
```

6. move your bot using teleop_twist_keyboard on seperate terminal
```
ros2 run teleop_twist_keyboard teleop_twist_keyboard
``` 
