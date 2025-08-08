# Investigating Potential of Social Navigation with Companion Robots in Outdoor Settings

## Overview
 Outdoor spaces and trails have played an important role in improving mental and physical health as well as fostering sustainable communities

 This project investigates the practicality of deploying robot companions to naturally accompany people walking outdoors

## Directions
Navigate to the directory where `requirements.txt` is and run `pip install -r requirements.txt`  
From there, there are two kinds of models that you can train:
1) A model that will predict N steps into the future of the person
2) A model that will predict the future position that the robot should navigate to  

To run #1, navigate to where `training.py` is and run the script. You can modify some of the parameters for training in `utils/config.json` for data augmentation and adjusting how many steps in the future you want to predict or how many steps in the past you want to take in as input. The model takes in N past positions and predicts the next M positions that the agent will be at. You can see the results by running `simulation.py`.  

To run #2, navigate to where `training_robot.py` is and run the script. You should only modify the number of past positions you want to take as input and the number of steps in the future where the robot should be. The model takes in N positions of the agent relative to the robot, and predicts a future position that's also relative to the robot. There are helper functions in `utils/data.py` to help you convert from relative to the robot frame to world space. Similar to before, you can run `simulation_with_robot.py` to see your results.


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
Thanks to [Zach Chavis](https://sites.google.com/umn.edu/zachsportfolio) and [Stephen J. Guy](https://www-users.cse.umn.edu/~sjguy/) for their invaluable mentorship during the NSF CSE REU at the University of Minnesota and making this experience possible
