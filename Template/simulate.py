import numpy as np
import math
import matplotlib.pyplot as plt
import json
import math
from Planner.local_planner import dwa_control, motion, plot_robot, plot_arrow
'''
Section 1: Load Obsticles
'''
# Load obstacle data from file
with open('map.txt', 'r') as file:
    map_data = json.load(file)

# Define constants
SIM_LOOP = 500
radius = 0.07
area = 1.5  # Set area size to center the plot on a larger grid

# Convert obstacle data to a list of obstacles with fixed radius
obstacle_list = [np.array((data['x'], data['y'], radius)) for data in map_data.values()]

'''
Section 2: Simulation Parameters
'''
class Config:
    """
    simulation parameter class
    """

    def __init__(self, obs_arr):
        # robot parameter
        self.max_speed = 1.0  # [m/s]
        self.min_speed = -0.5  # [m/s]
        self.max_yaw_rate = 40.0 * math.pi / 180.0  # [rad/s]
        self.max_accel = 0.2  # [m/ss]
        self.max_delta_yaw_rate = 40.0 * math.pi / 180.0  # [rad/ss]
        self.v_resolution = 0.01  # [m/s]
        self.yaw_rate_resolution = 0.1 * math.pi / 180.0  # [rad/s]
        self.dt = 0.1  # [s] Time tick for motion prediction
        self.predict_time = 3.0  # [s]
        self.to_goal_cost_gain = 0.15
        self.speed_cost_gain = 1
        self.obstacle_cost_gain = 0.05
        self.robot_stuck_flag_cons = 0.001  # constant to prevent robot stucked

        # if robot_type == RobotType.circle
        # Also used to check if goal is reached in both types
        self.robot_radius = 0.05  # [m] for collision check
        self.ob = obs_arr

config = Config(obstacle_list)

'''
Section X: Simulation Loop
'''
# Initial state of the robot [x, y, yaw, v, omega]
x = np.array([0.0, 0.0, math.pi / 8.0, 0.0, 0.0])

# Goal position
goal = np.array([1, 1])
show_animation = True
# Main simulation loop
for i in range(SIM_LOOP):
    print(f'Iteration: {i}')
    
    # Clear plot for each iteration
    plt.cla()

    # Check for ESC key to stop simulation
    plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])

    # Plot obstacles
    for obstacle in obstacle_list:
        ob_x, ob_y, ob_r = obstacle
        circle = plt.Circle((ob_x, ob_y), radius=ob_r, color='r', fill=False)
        plt.gca().add_patch(circle)

    # Run DWA control to get input (u) and predicted trajectory
    ob = np.array([[obs[0], obs[1]] for obs in obstacle_list])  # Convert obstacles to 2D list for DWA
    u, predicted_trajectory = dwa_control(x, config, goal, ob)

    # Update robot state based on motion model
    x = motion(x, u, config.dt)

    # Plot the predicted trajectory and robot position
    if show_animation:
        plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g")  # DWA predicted trajectory
        plt.plot(x[0], x[1], "xr")  # Current position of robot
        plt.plot(goal[0], goal[1], "xb")  # Goal position
        plot_robot(x[0], x[1], x[2], config)  # Plot the robot
        plot_arrow(x[0], x[1], x[2])  # Plot direction arrow

    # Set plot limits and grid
    plt.xlim(-area, area)
    plt.ylim(-area, area)
    plt.grid(True)
    plt.title(f"Iteration {i + 1}")
    plt.gca().set_aspect('equal', adjustable='box')

    # Check if goal is reached
    dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
    if dist_to_goal <= config.robot_radius:  # Assume robot radius or goal tolerance of 0.2 meters
        print("Goal reached!")
        break

    # Pause to simulate real-time update and display
    plt.pause(0.01)

print("Simulation finished.")

if show_animation:
    plt.show()

