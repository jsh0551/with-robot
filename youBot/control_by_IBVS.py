# Copyright 2024 @with-RL
#
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0

import argparse
import numpy as np
import math
import cv2
from youBot import YouBot
from scipy.optimize import fsolve
from scipy.spatial.transform import Rotation as R

SETTING = {
    'joint' : [0, -40, -65, -60, 0],
    't_joint' : [ -0,  -70.4, -55.4, -41.3,  -0. ],
    'gripper' : -0.05
}

TARGET = np.array([[124.72159624413145, 189.28028169014084], [122.72228750489619, 85.60360360360359], [165.925, 120.925], [202.3468520664349, 84.31324835843955], [199.7940492794049, 188.09995350999534], [166.11045828437133, 84.77634155895025]])
COLORS = {
    'red': [
        (np.array([0, 100, 100]), np.array([10, 255, 255]))
        # (np.array([160, 100, 100]), np.array([180, 255, 255]))
    ],
    'orange': [
        (np.array([11, 100, 100]), np.array([25, 255, 255]))
    ],
    'yellow': [
        (np.array([26, 100, 100]), np.array([35, 255, 255]))
    ],
    'green': [
        (np.array([36, 100, 100]), np.array([80, 255, 255]))
    ],
    'blue': [
        (np.array([100, 100, 100]), np.array([130, 255, 255]))
    ],
    'cyan': [
        (np.array([81, 100, 100]), np.array([99, 255, 255]))
    ]
}
PI_HALF = np.pi / 2

# forward kinematics
def fk(thetas, params):
    j1, j2, j3, j4 = thetas[:4]
    j0, pc = params[:2]
    # 월드 기준 자동차
    dc = R.from_quat(pc[3:]).as_euler('xyz')[-1]
    TWC = np.array([
        [np.cos(dc), -np.sin(dc), 0, pc[0]],
        [np.sin(dc),  np.cos(dc), 0, pc[1]],
        [         0,           0, 1, pc[2]],
        [         0,           0, 0,     1]
    ])

    # 자동차 -> joint-0
    TC0 = np.array([ # 좌표이동 및 y축을 기준으로 90도 회전
        [1, 0, 0, 0.0],
        [0, 1, 0, 0.166],
        [0, 0, 1, 0.099],
        [0, 0, 0, 1]
    ]) @ np.array([
        [np.cos(j0), -np.sin(j0), 0, 0],
        [np.sin(j0),  np.cos(j0), 0, 0],
        [         0,           0, 1, 0],
        [         0,           0, 0, 1]
    ])
    TW0 = TWC @ TC0

    # joint-0 -> joint-1
    ay1 = PI_HALF
    T01 = np.array([ # 좌표이동 및 y축을 기준으로 90도 회전
        [ np.cos(ay1), 0, np.sin(ay1), 0.0],
        [           0, 1,           0, 0.033],
        [-np.sin(ay1), 0, np.cos(ay1), 0.147],
        [           0, 0,           0, 1]
    ]) @ np.array([ # z축을 기준으로 j1만큼 회전
        [np.cos(j1), -np.sin(j1), 0, 0],
        [np.sin(j1),  np.cos(j1), 0, 0],
        [         0,           0, 1, 0],
        [         0,           0, 0, 1]
    ])
    TW1 = TW0 @ T01

    # joint-1 -> joint-2
    T12 = np.array([ # 좌표이동, 회전 없음
        [1, 0, 0, -0.155],
        [0, 1, 0,  0.0],
        [0, 0, 1,  0.0],
        [0, 0, 0,  1]
    ]) @ np.array([ # z축을 기준으로 j2만큼 회전
        [np.cos(j2), -np.sin(j2), 0, 0],
        [np.sin(j2),  np.cos(j2), 0, 0],
        [         0,           0, 1, 0],
        [         0,           0, 0, 1]
    ])
    TW2 = TW1 @ T12

    # joint-2 -> joint-3
    T23 = np.array([ # 좌표이동, 회전 없음
        [1, 0, 0, -0.135],
        [0, 1, 0,  0.0],
        [0, 0, 1,  0.0],
        [0, 0, 0,  1]
    ]) @ np.array([ # z축을 기준으로 j3만큼 회전
        [np.cos(j3), -np.sin(j3), 0, 0],
        [np.sin(j3),  np.cos(j3), 0, 0],
        [         0,           0, 1, 0],
        [         0,           0, 0, 1]
    ])
    TW3 = TW2 @ T23

    # joint-3 -> joint-4
    ay4 = - PI_HALF
    T34 = np.array([ # 좌표이동 및 y축을 기준으로 -90도 회전
        [ np.cos(ay4), 0, np.sin(ay4), -0.081],
        [           0, 1,           0,  0.0],
        [-np.sin(ay4), 0, np.cos(ay4),  0.0],
        [           0,  0,          0,  1]
    ]) @ np.array([ # z축을 기준으로 j4만큼 회전
        [np.cos(j4), -np.sin(j4), 0, 0],
        [np.sin(j4),  np.cos(j4), 0, 0],
        [         0,           0, 1, 0],
        [         0,           0, 0, 1]
    ])
    TW4 = TW3 @ T34

    pe_hat = TW4 @ np.array([ 0.0,   0.0,   0.123, 1])

    return pe_hat[:3]

def ik(thetas, params):
    pt = params[-1][:3]
    pe_hat = fk(thetas, params)
    # theta 범위 검증
    # if thetas[0] < np.deg2rad(-90) or np.deg2rad(75) < thetas[0]:
    #     return 10, 0, 0, 0
    # elif thetas[1] < np.deg2rad(-131.00) or np.deg2rad(131.00) < thetas[1]:
    #     return 10, 0, 0, 0
    # elif thetas[2] < np.deg2rad(-102.00) or np.deg2rad(102.00) < thetas[2]:
    #     return 10, 0, 0, 0
    # elif thetas[3] < np.deg2rad(-90.00) or np.deg2rad(90.00) < thetas[3]:
    #     return 10, 0, 0, 0
    return np.linalg.norm(pe_hat - pt), 0, 0, 0


class VisualServoBot(YouBot):
    def __init__(self):
        super().__init__()
        self.f_len = self.getGlobalFocalLength()
        dummyHandle = self.sim.createDummy(1.0)

        self.joints  = []
        for i,arm in enumerate(self.arms):
            jPos = SETTING['joint'][i]
            self.sim.setJointPosition(arm, np.radians(jPos))
        # Gripper Joint
        self.sim.setJointPosition(self.gripper, SETTING['gripper'])
        self.joints += self.arms
        self.joints.append(self.gripper)

        self.target = self.sim.getObject(f"/pe_ref")
        ee_position = self.sim.getObjectPosition(self.target)
        ee_angle = self.sim.getObjectOrientation(self.target) # radian
        self.targetPose = ee_position + ee_angle
        # target_thetas = self.solve(js, self.targetPose)
        # joint 제어 모드 변경
        for i,joint in enumerate(self.joints):
            self.sim.setObjectInt32Param(
                joint,
                self.sim.jointintparam_dynctrlmode,
                self.sim.jointdynctrl_position,
            )

    def run_step(self, count):
        # car control
        # self.control_car()
        # # arm control
        # self.control_arm()
        # # arm gripper
        # self.control_gripper()
        img = self.read_camera_1()
        bgr_img = img[::-1,:,::-1]
        for i,joint in enumerate(self.joints[:5]):
            jPos = self.sim.getJointPosition(joint)
            self.sim.setJointTargetPosition(joint, jPos)
        circle_img, self.ballPixelPositions = self.detect_features(bgr_img)
        print(self.ballPixelPositions)
        self.actuation(self.ballPixelPositions)
        # for i,joint in enumerate(self.joints[:5]):
        #     jPos = SETTING['t_joint'][i]
        #     self.sim.setJointTargetPosition(joint, np.radians(jPos))

    def actuation(self, ballPixelPositions):
        J = self.CalculateJacobian(ballPixelPositions)
        print(ballPixelPositions)
        if len(J) > 0:
            velPixel = (TARGET - np.array(ballPixelPositions)).flatten()
            Jpinv = np.linalg.pinv(J)
            velCam = Jpinv@velPixel
            x, y, z = velCam[:3]/2000
            a, b, c =velCam[3:]/10000
            print("pos!: ",self.targetPose[:3])
            print("angle!: ",np.round(np.degrees(self.targetPose[3:]),2))
            self.targetPose[0] += x
            self.targetPose[1] -= y
            self.targetPose[2] -= z
            self.targetPose[3] += a
            self.targetPose[4] += b
            self.targetPose[5] += c
            print("target pos!: ",self.targetPose[:3])
            print("target angle!: ",np.round(np.degrees(self.targetPose[3:]),2))
            js = self.read_joints(self.joints)
            ps = self.read_points(self.targetPose)
            target_thetas = self.solve(js, ps)
            print("target_thetas :",np.round(np.degrees(target_thetas),2))
            # for joint, theta in zip(self.joints, target_thetas):
            #     self.sim.setJointTargetPosition(joint, theta)
            diff_sum = self.trace_joint(self.joints, target_thetas)
            print("diff",diff_sum)

    def read_joints(self, joints):
        js = []
        for joint in joints:
            j = self.sim.getJointPosition(joint)
            js.append(j)
        return js

    def read_points(self, eePose):
        points = []
        # Car 위치
        points.append(self.sim.getObject(f"/youBot_ref"))
        # Joint-0 위치
        points.append(self.sim.getObject(f"/p0_ref"))
        # End Effector 위치
        points.append(self.sim.getObject(f"/pe_ref"))
        ps = []
        for point in points:
            p = self.sim.getObjectPosition(point)
            o = self.sim.getObjectQuaternion(point)
            # print(p,o)
            ps.append(np.array(p + o))
        # target
        xyz = eePose[:3]
        quat = list(R.from_euler('xyz', eePose[3:]).as_quat())
        ps.append(np.array(xyz + quat))
        return ps

    def solve(self, js, ps):
        p0, pt = ps[1], ps[-1]
        diff = pt[:2] - p0[:2]
        angle = math.atan2(diff[1], diff[0])
        d0 = R.from_quat(p0[3:]).as_euler('xyz')[-1]
        j0 = angle - d0 - PI_HALF
        target_thetas = fsolve(
            ik,
            [js[1], js[2], js[3], js[4]],
            [j0, ps[0], ps[-1]]
        )
        dt = R.from_quat(pt[3:]).as_euler('xyz')[-1]
        j4_diff = (j0 + PI_HALF - dt) % PI_HALF
        if PI_HALF / 2 < j4_diff:
            j4_diff -= PI_HALF
        elif j4_diff < -PI_HALF / 2:
            j4_diff += PI_HALF
        # target_thetas[3] += j4_diff
        return np.concatenate((np.array([j0]), target_thetas))

    def trace_joint(self, joints, target_thetas):
        js = self.read_joints(joints)
        diff_sum = 0
        thetas = []
        for i, target in enumerate(target_thetas):
            diff = js[i] - target
            if diff > 0.01:
                diff = 0.01
            elif diff < -0.01:
                diff = -0.01
            diff_sum += abs(diff)
            thetas.append(diff)
            # thetas.append(js[i] - min(0.02, max(-0.02, diff)))
        for joint, theta in zip(joints, target_thetas):
            self.sim.setJointTargetPosition(joint, theta)
        return diff_sum

    def getGlobalFocalLength(self):
        res, perspAngle = self.sim.getObjectFloatParameter(self.camera_1, self.sim.visionfloatparam_perspective_angle)
        res, resolution = self.sim.getVisionSensorResolution(self.camera_1)
        # distance per pixel
        planeWidth = 2 * math.tan(perspAngle / 2)
        distancePerPixel = planeWidth / resolution
        # global focal length
        pixelFocalLength = (resolution / 2) / math.tan(perspAngle / 2)
        globalFocalLength = pixelFocalLength * distancePerPixel
        return globalFocalLength

    def detect_features(self, image):
        # BGR에서 HSV로 변환
        image = image.copy()
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        ballPixelPositions = []
        for color, ranges in COLORS.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask += cv2.inRange(hsv, lower, upper)
        
            # 노이즈 제거
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # 윤곽선 찾기
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                for contour in contours:
                    M = cv2.moments(contour)
                    # 중심 좌표 계산
                    if M["m00"] != 0:
                        cX = (M["m10"] / M["m00"])
                        cY = (M["m01"] / M["m00"])
                        # 원 그리기
                        ballPixelPositions.append([cX, cY])
                        cv2.circle(image, (int(cX), int(cY)), 3, (255, 255, 0), -1)
        return image, ballPixelPositions
        
    def getImageJacobian(self, z, u, v):
        img_jacobian = np.array([[-self.f_len/z, 0, u/z, u*v/self.f_len, -self.f_len - u**2/self.f_len, v],
            [0, -self.f_len/z, v/z, self.f_len + v**2/self.f_len, -u*v/self.f_len, -u]])
        return img_jacobian

    def CalculateJacobian(self, ballPixelPositions):
        ee = self.sim.getObject(f"/pe_ref")
        x, y, z = self.sim.getObjectPosition(ee)
        img_jacobians = []
        for position,t_position in zip(ballPixelPositions, TARGET):
            u,v = position
            tu,tv = t_position
            # s = np.linalg.norm(np.array(position) - np.array(t_position))/256
            img_jacobian = self.getImageJacobian(z, u, v)
            img_jacobians.append(img_jacobian)
        if img_jacobians:
            img_jacobians = np.concatenate(img_jacobians, axis=0)
        return img_jacobians

    def show_video(self):
        if self.viz == True:
            img = self.read_camera_1()
            if self.version != 0:
                rgb_img = img[::-1,:,:]
                depth = self.get_relative_depth(rgb_img)
                cv2.imshow("object", depth)
            else:
                bgr_img = img[::-1,:,::-1]
                circle_img, self.ballPixelPositions = self.detect_features(bgr_img)
                # print(self.ballPixelPositions)
                cv2.imshow("object", circle_img)
            cv2.waitKey(10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('---encoder', type=str, default='vitb', choices=['vitb', 'vitl'])
    parser.add_argument('---viz', type=bool, default=False, choices=[False, True])
    parser.add_argument('--version', type=int, default=0, choices=[0, 1, 2])
    args = parser.parse_args()

    client = VisualServoBot()
    client.init_coppelia(args.encoder, args.viz, args.version)
    client.run_coppelia()
