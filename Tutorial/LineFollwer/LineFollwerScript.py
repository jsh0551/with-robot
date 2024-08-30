import time
import numpy as np
import cv2
from pynput import keyboard
from pynput.keyboard import Key, Listener
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

class LineFollwer:
    def __init__(self):
        client = RemoteAPIClient()
        self.sim = client.require('sim')
        self.sync = True
        self.quit = False
        Listener(on_press=self.on_press).start()
        # coppelia sim init
        self.sim.setStepping(self.sync)
        self.leftjoint = self.sim.getObject('/DynamicLeftJoint')
        self.rightjoint = self.sim.getObject('/DynamicRightJoint')
        self.leftsensor = self.sim.getObject('/LeftSensor')
        self.rightsensor = self.sim.getObject('/RightSensor')
        self.img = None
        self.left_intensity = 0.0
        self.right_intensity = 0.0

    def set_sync(self, flag):
        self.sync = flag
        self.sim.setStepping(self.sync)

    def step(self, dt = 1):
        if self.sync:
            self.sim.step()
        else:
            time.sleep(dt)

    def stream_image(self, VIZ):
        if VIZ:
            cv2.imshow("object", self.img)
            cv2.waitKey(10)

    def on_press(self, key):
        if key == keyboard.KeyCode.from_char("q"):
            self.quit = True

    def actuation(self):
        self.sim.setJointTargetVelocity(self.leftjoint, 2)
        self.sim.setJointTargetVelocity(self.rightjoint, 2)
        # turn to intense side
        if self.right_intensity > 0.5 and self.left_intensity < 0.5:
            print('turn left')
            self.sim.setJointTargetVelocity(self.rightjoint, 4)
        elif self.right_intensity < 0.5 and self.left_intensity > 0.5:
            print('turn right')
            self.sim.setJointTargetVelocity(self.leftjoint, 4)

    def sensing(self):
        leftVisionResult = self.sim.readVisionSensor(self.leftsensor)
        rightVisionResult = self.sim.readVisionSensor(self.rightsensor)
        depth, resolution = self.sim.getVisionSensorImg(self.leftsensor)
        uint8Numbers = self.sim.unpackUInt8Table(depth)
        img = np.array(uint8Numbers, dtype=np.uint8).reshape((resolution[1], resolution[0], 3))
        # min : intensity, r, g, b, depth (0:5)
        # max : intensity, r, g, b, depth (5:10)
        # average : intensity, r, g, b, depth (10:15)
        if leftVisionResult == -1:
            leftVisionState, leftPacket1, leftPacket2 = -1, [], []
        else:
            leftVisionState, leftPacket1, leftPacket2 = leftVisionResult
            self.left_intensity = leftPacket1[10]
        if rightVisionResult == -1:
            rightVisionState, rightPacket1, rightPacket2 = -1, [], []
        else:
            rightVisionState, rightPacket1, rightPacket2 = rightVisionResult
            self.right_intensity = rightPacket1[10]
        img = img[::-1,:, ::-1]
        self.img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)
        self.step()

if __name__ == "__main__":
    follwer = LineFollwer()
    follwer.sim.startSimulation()
    follwer.set_sync(True)
    while not follwer.quit:
        follwer.actuation()
        follwer.sensing()
        follwer.stream_image(True)

    follwer.sim.stopSimulation()