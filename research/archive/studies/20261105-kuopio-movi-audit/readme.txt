KUOPIO GAIT DATASET

This dataset contains motion capture (3D marker trajectories, ground reaction forces and moments), inertial measurement unit (wearable Movella Xsens MTw Awinda sensors on the pelvis, both thighs, both shanks, and both feet), and sagittal-plane video (anatomical keypoints identified with the OpenPose human pose estimation algorithm) data.
The data is from 51 willing participants and collected in the HUMEA laboratory in the University of Eastern Finland, Kuopio, Finland, between 2022 and 2023.

The file structure contains an Excel file containing information of the participants, data folders under each subject (numbered 01 to 51), and a MATLAB script.

The Excel file has the following data for the participants:
	- ID: ID of the participants from 1 to 51
	- Age: age of the participant in years
	- Gender: biological sex as M for male, F for female
	- Leg: the participant's dominant leg, identified by asking which foot the participant would use to kick a football; R for right, L for left
	- Height: height of the participant in centimeters
	- Invalid_trials: list of invalid trials in the motion capture data (MOCAP) data, usually classified as such because the participant did not properly step on the middle force plate
	- IAD: inter-asis distance in millimeters, the distance between palpated left and right anterior superior iliac spine, measured with a caliper
	- Left_knee_width: width of the left knee from medial epicondyle to lateral epicondyle in millimeters, palpated and measured with a caliper
	- Right_knee_width: same as above for the right knee
	- Left_ankle width: width of the left ankle from medial malleolus to lateral malleolus in millimeters, palpated and measured with a caliper
	- Right_ankle_width: same as above for the right ankle
	- Left_thigh_length: the distance between the greater trochanter of the left femur and the lateral epicondyle of the left femur in millimeters, palpated and measured with a measuring tape
	- Right_thigh_length: same as above for the right thigh
	- Left_shank_length: the distance between the medial epicondyle of the femur and the medial malleolus of the tibia in millimeters, palpated and measured with a measuring tape
	- Right_shank_length: same as above for the right shank
	- Mass: mass in kilograms, measured on a force plate just before the walking measurements
	- ICD: inter-condylar distance of the knee of the dominant leg, measured from low-field MRI
	- Left_knee_width_mocap: distance between reflective MOCAP markers on the medial and lateral epicondyles of the knee in millimeters, measured from a static standing trial; -1 for missing (subject did not have those markers)
	- Right_knee_width_mocap: same as above for the right knee

The folders under each subject (folders numbered 01 to 51) are as follows:
    - imu: "Raw" inertial measurement unit (IMU) data files that can be read with Xsens Device API (included in Xsens MT Manager 4.6, which may be unavailable these days, not sure). You won't need this if you use the data in the imu_extracted folder.
    - imu_extracted: IMU data extracted from those data files using the Xsens Device API, so you don't have to.
        - The data is saved as MATLAB structs where the fields are named as a sensor ID (e.g., "B42D48"). The sensor IDs and their corresponding IMU locations are as follows:
            - pelvis IMU: B42DA3
            - right femur IMU: B42DA2
            - left femur IMU: B42D4D
            - right tibia IMU: B42DAE
            - left tibia IMU: B42D53
            - right foot IMU: B42D48
            - left foot IMU: B42D51 (except for subjects 01 and 02, where left foot IMU has the ID B42D4E)
        - Some of the data are just zeros as they couldn't be read from these sensors, but under each sensor, the fields "calibratedAcceleration", "freeAcceleration", "time", "rotationMatrix", and "quaternion" contain usable data.
            - time: Contains time stamps of the measurement at each frame recorded at 100 Hz, so if you remove the first value from all values in the time vector and divide the result by 100, you will get the time in seconds from the beginning of the walking trial.
            - calibratedAcceleration and freeAcceleration: Contain triaxial acceleration data from the accelerometers of the IMU. freeAcceleration is just calibratedAcceleration without the effect of Earth's gravitational acceleration.
            - rotationMatrix: Orientations of the IMU as rotation matrices.
            - quaternion: Orientations of the IMU as quaternions.
    - openpose: Trajectories of the keypoints identified from sagittal plane video frames, saved as json files.
        - The keypoints are from the BODY_25 model of OpenPose (https://cmu-perceptual-computing-lab.github.io/openpose/web/html/doc/md_doc_02_output.html).
        - Each frame in the video has its own json file.
        - You can use the function in the script "OpenPose_to_keypoint_table.m" in the root folder to read the keypoint trajectories and confidences of all frames in a walking trial into MATLAB tables. The function takes as argument the path to the folder containing the json files of the walking trial.
	- Note that some subjects (11, 14, 37, 49) do not have keypoint and IMU data.
The folders under each subject are divided into three ZIP archives with 17 subjects each.

The script "OpenPose_to_keypoint_table.m" is a MATLAB script for extracting keypoint trajectories and confidences from JSON files into tables in MATLAB.


Publication pending.


Contact: Jere Lavikainen, jere.lavikainen@uef.fi
