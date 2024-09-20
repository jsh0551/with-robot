# Copyright 2024 @with-RL
#
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0

import argparse
from youBot import YouBot

class KeyboardBot(YouBot):
    def run_step(self, count):
        # car control
        self.control_car()
        # arm control
        self.control_arm()
        # arm gripper
        self.control_gripper()
        # read lidarqa
        self.read_lidars()
        # read camera
        self.read_camera_1()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('---encoder', type=str, default='vitb', choices=['vitb', 'vitl'])
    parser.add_argument('--version', type=int, default=1, choices=[1, 2])
    args = parser.parse_args()

    client = KeyboardBot()
    client.init_coppelia(args.encoder, args.version)
    client.run_coppelia()
