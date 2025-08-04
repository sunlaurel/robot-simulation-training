# Investigating Potential of Social Navigation with Companion Robots in Outdoor Settings

## Overview
Trained a neural network to walk alongside people in simulation

## Sample Results
We trained a multilayer perceptron (MLP) on crowd data of people walking and generated robot trajectories, passing in past relative vectors between the robot's past trajectory and the person's past trajectory to predict the future position that the robot should walk to in the robot's frame. Below are some sample trajectories and the predicted positions outputted by the model.

<html>
  <! :-------------------------:|:-------------------------: -->
  <! ![Sample Trajectory #1](/images/traj1.png)  |  ![Sample Trajectory #2](/images/traj2.png)  |  ![Sample Trajectory #3](/images/traj3.png) --> 

  <p float="center">
    <img src="/images/traj1.png" width="250" />
    <img src="/images/traj2.png" width="250" /> 
    <img src="/images/traj3.png" width="250" />
  </p>

</html>





## Acknowledgements
Thanks to (Zach Chavis)["chavi014@umn.edu"] and (Stephen J. Guy)[https://www-users.cse.umn.edu/~sjguy/] for their invaluable mentorship during the NSF CSE REU at the University of Minnesota and making this experience possible
