import time
import numpy as np
import cv2
from pynput import keyboard
from pynput.keyboard import Key, Listener
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

class Base:
    def __init__(self):
        client = RemoteAPIClient()
        self.sim = client.require('sim')
        self.sync = True
        self.quit = False
        Listener(on_press=self.on_press).start()
        # coppelia sim init
        self.sim.setStepping(self.sync)

    def set_sync(self, flag):
        self.sync = flag
        self.sim.setStepping(self.sync)

    def step(self, dt = 1):
        if self.sync:
            self.sim.step()
        else:
            time.sleep(dt)

    def on_press(self, key):
        if key == keyboard.KeyCode.from_char("q"):
            self.quit = True
        elif key == keyboard.KeyCode.from_char("w"):
            self.quit = True