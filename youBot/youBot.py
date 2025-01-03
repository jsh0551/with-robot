# Copyright 2024 @with-RL
#
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import dataclass
from abc import abstractmethod
import numpy as np
import traceback
from pynput import keyboard
from pynput.keyboard import Key, Listener

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

import cv2
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm

from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = Compose([
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

@dataclass
class Control:
    vel_X: float = 0
    vel_Y: float = 0
    vel_Z: float = 0
    arm_0: float = 0
    arm_1: float = 0
    arm_2: float = -np.pi / 4
    arm_3: float = -np.pi / 4
    arm_4: float = 0
    gripper: float = -0.04


class YouBot:
    def __init__(self):
        self.client = RemoteAPIClient()
        self.sim = self.client.require("sim")
        self.run_flag = True
        self.viz = False
        self.version = 0
        self.control = Control()
        self.youBot_ref = self.sim.getObject("/youBot_ref")
        # Wheel Joints: front left, rear left, rear right, front right
        self.wheels = []
        self.wheels.append(self.sim.getObject("/rollingJoint_fl"))
        self.wheels.append(self.sim.getObject("/rollingJoint_rl"))
        self.wheels.append(self.sim.getObject("/rollingJoint_fr"))
        self.wheels.append(self.sim.getObject("/rollingJoint_rr"))
        # Arm Joints
        self.arms = []
        for i in range(5):
            arm = self.sim.getObject(f"/youBotArmJoint{i}")
            self.arms.append(arm)
        # Gripper Joint
        self.gripper = self.sim.getObject(f"/youBotGripperJoint2")
        # lidar
        self.lidars = []
        for i in range(1, 14):
            self.lidars.append(self.sim.getObject(f"/lidar_{i:02d}"))
        # camera
        self.camera_1 = self.sim.getObject(f"/camera_1")


    def on_press(self, key):
        deltaX, deltaZ = 1.0, np.pi / 10
        if key == Key.up:
            self.control.vel_X += deltaX
            self.control.vel_Z += min(deltaZ, abs(self.control.vel_Z)) * (
                -1 if self.control.vel_Z > 0 else 1
            )
        if key == Key.down:
            self.control.vel_X -= deltaX
            self.control.vel_Z += min(deltaZ, abs(self.control.vel_Z)) * (
                -1 if self.control.vel_Z > 0 else 1
            )
        if key == Key.left:
            self.control.vel_X += min(deltaX, abs(self.control.vel_X)) * (
                -1 if self.control.vel_X > 0 else 1
            )
            self.control.vel_Z += deltaZ
        if key == Key.right:
            self.control.vel_X += min(deltaX, abs(self.control.vel_X)) * (
                -1 if self.control.vel_X > 0 else 1
            )
            self.control.vel_Z -= deltaZ
        self.control.vel_X = min(max(self.control.vel_X, -20), 20)
        self.control.vel_Y = 0
        self.control.vel_Z = min(max(self.control.vel_Z, -np.pi), np.pi)

        delta = 0.1
        if key == keyboard.KeyCode.from_char("w"):
            self.control.arm_0 += delta
        if key == keyboard.KeyCode.from_char("s"):
            self.control.arm_0 -= delta
        self.control.arm_0 = min(max(self.control.arm_0, -np.pi), np.pi)

        if key == keyboard.KeyCode.from_char("e"):
            self.control.arm_1 += delta
        if key == keyboard.KeyCode.from_char("d"):
            self.control.arm_1 -= delta
        self.control.arm_1 = min(max(self.control.arm_1, -np.pi), np.pi)

        if key == keyboard.KeyCode.from_char("r"):
            self.control.arm_2 += delta
        if key == keyboard.KeyCode.from_char("f"):
            self.control.arm_2 -= delta
        self.control.arm_2 = min(max(self.control.arm_2, -np.pi), np.pi)

        if key == keyboard.KeyCode.from_char("t"):
            self.control.arm_3 += delta
        if key == keyboard.KeyCode.from_char("g"):
            self.control.arm_3 -= delta
        self.control.arm_3 = min(max(self.control.arm_3, -np.pi), np.pi)

        if key == keyboard.KeyCode.from_char("y"):
            self.control.arm_4 += delta
        if key == keyboard.KeyCode.from_char("h"):
            self.control.arm_4 -= delta
        self.control.arm_4 = min(max(self.control.arm_4, -np.pi), np.pi)

        delta = 0.01
        if key == keyboard.KeyCode.from_char("u"):
            self.control.gripper += delta
        if key == keyboard.KeyCode.from_char("j"):
            self.control.gripper -= delta
        self.control.gripper = min(max(self.control.gripper, -0.04), 0.04)

        if key == keyboard.KeyCode.from_char("q"):
            self.run_flag = False
        if key == keyboard.KeyCode.from_char("a"):
            self.control.vel_X = 0
            self.control.vel_Z = 0

    def init_coppelia(self, encoder, viz, version):
        # load RDE
        self.viz = viz
        self.version = version
        if version == 0:
            pass
        if version == 1:
            from depth_anything.dpt import DepthAnything
            self.depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoder)).to(DEVICE).eval()
        elif version == 2:
            from depth_anything_v2.dpt import DepthAnythingV2
            self.depth_anything = DepthAnythingV2(**model_configs[encoder])
            self.depth_anything.load_state_dict(torch.load(f'../checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
            self.depth_anything = self.depth_anything.to(DEVICE).eval()
        # reference

    def control_car(self):
        self.sim.setJointTargetVelocity(
            self.wheels[0],
            -self.control.vel_X + self.control.vel_Z,
        )
        self.sim.setJointTargetVelocity(
            self.wheels[1],
            -self.control.vel_X + self.control.vel_Z,
        )
        self.sim.setJointTargetVelocity(
            self.wheels[2],
            -self.control.vel_X - self.control.vel_Z,
        )
        self.sim.setJointTargetVelocity(
            self.wheels[3],
            -self.control.vel_X - self.control.vel_Z,
        )

    def control_arm(self):
        self.sim.setJointTargetPosition(self.arms[0], self.control.arm_0)
        self.sim.setJointTargetPosition(self.arms[1], self.control.arm_1)
        self.sim.setJointTargetPosition(self.arms[2], self.control.arm_2)
        self.sim.setJointTargetPosition(self.arms[3], self.control.arm_3)
        self.sim.setJointTargetPosition(self.arms[4], self.control.arm_4)

    def control_gripper(self):
        self.sim.setJointTargetVelocity(self.gripper, self.control.gripper)

    def read_lidars(self):
        scan = []
        for id in self.lidars:
            scan.append(self.sim.readProximitySensor(id))
        return scan

    def read_camera_1(self):
        result = self.sim.getVisionSensorImg(self.camera_1)
        img = np.frombuffer(result[0], dtype=np.uint8)
        img = img.reshape((result[1][1], result[1][0], 3))
        return img

    def show_video(self):
        if self.viz == True:
            img = self.read_camera_1()
            if self.version != 0:
                rgb_img = img[::-1,:,:]
                depth = self.get_relative_depth(rgb_img)
                cv2.imshow("object", depth)
            else:
                img = img[::-1,:, ::-1]
                rgb_img = img[::-1,:,:]
                cv2.imshow("object", rgb_img)
            cv2.waitKey(10)

    def get_relative_depth(self, rgb_img):
        image = rgb_img / 255.0
        h, w = image.shape[:2]
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            depth = self.depth_anything(image)

        depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.cpu().numpy().astype(np.uint8)

        # gray scale
        depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        return depth

    def run_coppelia(self):
        # key input
        Listener(on_press=self.on_press).start()
        # start simulation
        self.sim.setStepping(True)
        self.sim.startSimulation()
        count = 0
        while self.run_flag:
            count += 1
            # step
            try:
                self.run_step(count)
                self.sim.step()
                self.show_video()
            except Exception as e:
                print(e)
                tb = traceback.format_exc()
                print("Traceback details:")
                print(tb)
                break
        cv2.destroyAllWindows()
        self.sim.stopSimulation()

    @abstractmethod
    def run_step(self, count):
        pass
