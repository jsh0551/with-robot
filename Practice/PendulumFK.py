import time
import numpy as np
from pynput import keyboard
from ModelBase import PendulumBase

THETA1 = -30
THETA2 = 120


class Pendulum(PendulumBase):
    def __init__(self):
        super().__init__()
        # coppelia sim init
        self.joint1 = self.sim.getObject('/Joint')
        self.arm1 = self.sim.getObject('/Cuboid')
        self.joint2 = self.sim.getObject('/Joint2')
        self.arm2 = self.sim.getObject('/Cuboid2')
        self.endpoint = self.sim.getObject('/EndPoint')
        self.targetpoint = self.sim.getObject('/TargetPoint')
        self.sim.setJointMode(self.joint1, self.sim.jointmode_dynamic, self.sim.jointdynctrl_position)
        self.sim.setJointMode(self.joint2, self.sim.jointmode_dynamic, self.sim.jointdynctrl_position)
        self.sim.setJointPosition(self.joint1, np.radians(THETA1))
        self.sim.setJointPosition(self.joint2, np.radians(THETA2))
        joint1_pos = self.sim.getObjectPosition(self.joint1)
        joint2_pos = self.sim.getObjectPosition(self.joint2)
        end_pos = self.sim.getObjectPosition(self.endpoint)
        self.l1 = np.linalg.norm(np.array(joint1_pos) - np.array(joint2_pos))
        self.l2 = np.linalg.norm(np.array(joint2_pos) - np.array(end_pos))
        self.Q2 = np.array([0, 0, self.l2, 1])
        self.orin = np.array([0, 0, 1.2, 1])
        self.r1 = THETA1/180
        self.r2 = THETA2/180
        self.flag1, self.flag2 = False, False
        self.delta1, self.delta2 = 0.005, 0.005
        self.reset_target()

    def get_distance(self, point):
        return np.linalg.norm(self.orin[:3] - point[:3])
    
    def get_angle(self, point):
        y = point[1] - self.orin[1]
        z = point[2] - self.orin[2]
        # tmp = np.arctan(y/z)
        tmp = np.arctan2(y, z)
        return tmp

    def reset_target(self):
        d = np.random.uniform(0.51, 1.05)
        theta =  np.random.uniform(0, np.pi*(117/180))
        self.ty = d*np.sin(theta)
        self.tz = d*np.cos(theta) + self.orin[2]
        self.tdist = self.get_distance(np.array([0, self.ty, self.tz]))
        self.ttheta = self.get_angle([0, self.ty, self.tz])
        self.sim.setObjectPosition(self.targetpoint, [0.1, self.ty, self.tz])

    def actuation(self):
        ttheta1 = -np.pi*self.r1
        ttheta2 = np.pi*self.r2
        self.sim.setJointTargetPosition(self.joint1, ttheta1)
        self.sim.setJointTargetPosition(self.joint2, ttheta2)

    def sensing(self):
        joint1_pos = self.sim.getObjectPosition(self.joint1)
        joint1_angle = self.sim.getJointPosition(self.joint1)
        joint2_angle = self.sim.getJointPosition(self.joint2) - np.pi
        H01 = np.array(
                    [
                        [1, 0, 0, 0],
                        [0, np.cos(joint1_angle), -np.sin(joint1_angle), joint1_pos[1]],
                        [0, np.sin(joint1_angle), np.cos(joint1_angle), joint1_pos[2]],
                        [0, 0, 0, 1],
                    ]
        )
        H12 = np.array(
                    [
                        [1, 0, 0, 0],
                        [0, np.cos(joint2_angle), -np.sin(joint2_angle), 0],
                        [0, np.sin(joint2_angle), np.cos(joint2_angle), self.l1],
                        [0, 0, 0, 1],
                    ]
        )
        H02 = H01@H12
        Q0 = H02@self.Q2
        # calculate distance and angle
        D = self.get_distance(Q0)
        ANGLE = self.get_angle(Q0)
        if abs(D - self.tdist) < self.delta2:
            self.flag2 = True
            pass
        elif D > self.tdist:
            self.r2 -= self.delta2
        elif D < self.tdist:
            self.r2 += self.delta2
        if abs(ANGLE - self.ttheta) < np.pi*self.delta1:
            self.flag1 = True
            pass
        elif ANGLE > self.ttheta:
            self.r1 -= self.delta1
        elif ANGLE < self.ttheta:
            self.r1 += self.delta1
        self.r1 = np.clip(self.r1, -63/180, 1 - 63/180)
        self.r2 = np.clip(self.r2, 55/180, 1)
        print("*"*60)
        print(f'D : {D}, target : {self.tdist}, error : {abs(D - self.tdist)}')
        print(f'ANGLE : {np.degrees(ANGLE)}, target : {np.degrees(self.ttheta)}, error : {abs(ANGLE - self.ttheta)}')
        print(f'r1 : {self.r1}, r2 : {self.r2}')
        self.step()

if __name__ == "__main__":
    follwer = Pendulum()
    follwer.sim.startSimulation()
    follwer.set_sync(True)
    while not follwer.quit:
        follwer.actuation()
        follwer.sensing()
        if follwer.flag1 and follwer.flag2:
            follwer.reset_target()
            follwer.flag1 = False
            follwer.flag2 = False
    follwer.sim.stopSimulation()