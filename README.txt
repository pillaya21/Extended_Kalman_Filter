
Extended Kalman Filter to estimate the position of a vehicle along a trajectory using available measurements and a motion model.

This script was created as a submission to the EKF project on State Estimation and Localization for self driving cars course on Coursera

The vehicle is equipped with a very simple type of LIDAR sensor, which returns range and bearing measurements corresponding to individual landmarks in the environment. The global positions of the landmarks are assumed to be known beforehand. We will also assume known data association, that is, which measurment belong to which landmark.

Motion Model
The vehicle motion model recieves linear and angular velocity odometry readings as inputs, and outputs the state (i.e., the 2D pose) of the vehicle.The process noise has zero mean normal distribution with a constant covariance Q

Measurement model

the measurement model relates the current pose of the vehicle to the LIDAR range and bearing measurements. The landmark measurements noise has a zero mean normal distribution with a constant covariance R

