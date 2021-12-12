import airsim
import cv2
import numpy as np
import os, sys, time

from msgpackrpc.future import Future
import pprint

import utils

# Use below in settings.json with Blocks environment
"""
{
	"SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/master/docs/settings.md",
	"SettingsVersion": 1.2,
	"SimMode": "Multirotor",
	"ClockSpeed": 1,
	
	"Vehicles": {
		"Drone1": {
		  "VehicleType": "SimpleFlight",
		  "X": 4, "Y": 0, "Z": -2
		},
		"Drone2": {
		  "VehicleType": "SimpleFlight",
		  "X": 8, "Y": 0, "Z": -2
		}

    }
}
"""

if (len(sys.argv) > 1):
	drone_name = sys.argv[1]
else:
	drone_name = input("Enter drone name:")

# connect to the AirSim simulator
client = airsim.MultirotorClient(ip='172.17.167.208')
client.confirmConnection()
client.enableApiControl(True, drone_name)
client.armDisarm(True, drone_name)

# airsim.wait_key('Press any key to takeoff')
f1 = client.takeoffAsync(vehicle_name=drone_name)
f1.join()


state1 = client.getMultirotorState(vehicle_name=drone_name)
s = pprint.pformat(state1)
print("state: %s" % s)

# airsim.wait_key('Press any key to move vehicles')
f1 = client.moveToPositionAsync(-5, 5, -10, 5, vehicle_name=drone_name)
f1.join()

# airsim.wait_key('Press any key to take images')

while (True):
    # try:
    cam_color = utils.capture_as_np(client, drone_name)
    cv2.imshow(drone_name, cam_color)
    # except:
    #     print('error, cam is ', cam_color)
    #     break
    if (cv2.waitKey(1) == 27): # ESC
        break



client.armDisarm(False, drone_name)
client.reset()

# that's enough fun for now. let's quit cleanly
client.enableApiControl(False, drone_name)


