#!/home/jetson/ros_venv/bin/python3
import asyncio
import websockets
import numpy as np
# from openpyxl import Workbook,load_workbook
import time
from geometry_msgs.msg import Twist
import rospy
import json


rospy.init_node("barrier_function", anonymous=True)
rospy.spin()
# sheet = load_workbook("barrier.xlsx")
# ws = sheet.active

# Parameters
# x_init = np.array([4.0, 8.0])  # Initial position
# x_obs = np.array([2.5, 2.5])  # Obstacle position
# r = 1.0  # Safe radius around the obstacle
# lambda_val = 1000.0  # Barrier weight
x_target = np.array([0.0, 0.0])  # Target position
alpha = 0.1  # Learning rate
max_iter = 1000  # Max iterations




cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10) # JetRacer Velocity Publisher



# def objective_gradient(x, x_target, x_obs, r, lambda_val):
def objective_gradient(x, x_target):
    # Compute the gradient of the total cost function
    grad_distance = 2 * (x - x_target)
    # grad_barrier = (2 * (x - x_obs)) / (np.linalg.norm(x - x_obs)**2 - r**2)
    # grad_total = grad_distance + lambda_val * grad_barrier
    grad_total = grad_distance

    return grad_total

# def gradient_descent(x_init, x_target, x_obs, r, lambda_val, alpha, max_iter):
def gradient_descent(x_init, x_target, alpha, max_iter):

    x_old = x_init
    # ws["A" + str(2)].value = x_old[0]
    # ws["B" + str(2)].value = x_old[1]
    start_time = time.time()
    for i in range(max_iter):
        time.sleep(0.1)
        grad = objective_gradient(x_old, x_target)
        x_new = x_old - alpha * grad  # Update position
        end_time = time.time()
        print(f"start {start_time} , end : {end_time}")
        vx_vy = (x_new - x_old)/(end_time - start_time)
        linear_velocity = np.linalg.norm(vx_vy)
        ang_Velocity = linear_velocity/30
        start_time = end_time
        x_old = x_new
        # print or track the cost to check convergence
        # cost = np.linalg.norm(x - x_target) ** 2 + lambda_val * (-np.log(np.linalg.norm(x - x_obs) ** 2 - r ** 2))
        cost = np.linalg.norm(x_new - x_target) ** 2
        print(f"Iteration {i + 1}, Cost: {cost}, Position: {x_new} , velocity: {linear_velocity} , ang_vel: {ang_Velocity}")
        # ws["A" + str(i + 3)].value =x_old[0]
        # ws["B" + str(i + 3)].value =x_old[1]

        # If the position is close enough to the target, stop the iteration
        if (np.linalg.norm(x_new - x_target) < 1e-6) or (linear_velocity < 0.01):
            break
    return [x_new, linear_velocity, ang_Velocity]


# Make sure to change IP addresss to match Laptop's IP Address
async def connect_to_mocap(server_ip="192.168.86.221"):
    uri = f"ws://{server_ip}:8765"


    while True:
        try:
            async with websockets.connect(uri) as websocket:
                print("Connected to motion capture system")
                while True:
                    try:
                        message = await websocket.recv()
                        data = json.loads(message)

                        if "Player" in data["objects"]:
                            x = data["objects"]["Player"]["x"]
                            y = data["objects"]["Player"]["y"]
                            print(f'x = {x}, y = {y}')

                            x_init = np.array([x, y])
                            final_position, linear_velocity, ang_Velocity = gradient_descent(x_init, x_target, alpha, max_iter)

                            cmd = Twist()
                            cmd.linear.x = linear_velocity
                            cmd.angular.z = ang_Velocity

                            # Publish linear & angular velocities to the car
                            cmd_vel_pub.publish(cmd)



                    except websockets.exceptions.ConnectionClosed:

                        break
        except Exception as e:
           print(f"Motion capture connection error: {e}")
        await asyncio.sleep(0.5)

# Start Motion Capture Thread
def start_mocap_client():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(connect_to_mocap())

# Run Motion Capture in a seperate thread
start_mocap_client()
# Run gradient descent to find the optimal path
# final_position = gradient_descent(x_init, x_target, alpha, max_iter)
# print(f"Final position: {final_position}")
# sheet.save("no_obstacles.xlsx")