# Copyright 2024 @with-RL
#
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0

import argparse
import numpy as np
import math
import cv2
import cv2.aruco as aruco
from youBot import YouBot
from scipy.optimize import fsolve
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize


SETTING = {
    'joint' : [0, -55, -60, -40, 0],
    # 'joint' : [0, 0, 0, 0, 0],
    # 'joint' : [ -0,  -70.4, -55.4, -41.3,  -0.],
    't_joint' : [ -0,  -70.4, -55.4, -41.3,  -0. ],
    'gripper' : -0.05
}
## aruco target
TARGET = np.array([[454.0, 291.0], [352.0, 292.0], [352.0, 192.0], [458.0, 190.0],
 [172.0, 432.0], [77.0, 433.0], [70.0, 340.0], [170.0, 335.0]
 ])
##

## feat 8
# TARGET = np.array([[124.0, 202.53735632183907], [92.8230890464933, 86.91725768321513], [87.7598797250859, 136.79725085910653], [200.8926619828259, 85.00468384074941], [198.43244506778865, 188.51706404862082], [164.55939226519337, 169.34622467771638], [200.05729820158928, 120.95650355499791], [157.65282865282865, 107.45706145706146]])
## feat 6
# TARGET = np.array([[124.0, 202.53735632183907], [92.8230890464933, 86.91725768321513], [87.7598797250859, 136.79725085910653], [200.8926619828259, 85.00468384074941], [198.43244506778865, 188.51706404862082], [157.65282865282865, 107.45706145706146]])
## feat 5
# TARGET = np.array([[123.86056253740274, 202.97366846199878], [92.3797729618163, 86.05572755417957], [87.30468319559228, 136.45509641873278], [201.46764705882353, 84.12352941176471], [199.05340078695895, 188.72625070264192]])
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
    # 'purple': [
    #     (np.array([131, 100, 100]), np.array([155, 255, 255]))
    # ],
    # 'pink': [
    #     (np.array([156, 100, 100]), np.array([165, 255, 255]))
    # ],
    # 'cyan': [
    #     (np.array([81, 100, 100]), np.array([99, 255, 255]))
    # ]
}
PI_HALF = np.pi / 2

# forward kinematics
def fk(thetas, pc):
    j0, j1, j2, j3, j4 = thetas
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

    # joint-4 -> camera_1
    theta_x = np.deg2rad(-20)   
    theta_y = np.deg2rad(-0.356) 
    theta_z = np.deg2rad(-179)  

    T_Cam = np.array([ # 좌표이동
        [1, 0, 0, 0.0],
        [0, 1, 0, -0.01966],
        [0, 0, 1, 0.07455],
        [0, 0, 0, 1]
    ])
    R_x = np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x), 0],
        [0, np.sin(theta_x), np.cos(theta_x), 0],
        [0, 0, 0, 1]
    ])
    R_y = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y), 0],
        [0, 1, 0, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y), 0],
        [0, 0, 0, 1]
    ])
    R_z = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0, 0],
        [np.sin(theta_z), np.cos(theta_z), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    T4CAM = T_Cam @ R_x @ R_y @ R_z
    TWCAM = TW4 @ T4CAM
    position = TWCAM[:3, 3]
    rotation_matrix = TWCAM[:3, :3]
    rpy = -R.from_matrix(rotation_matrix).as_euler('xyz', degrees=True)
    return position, rpy

def ik(thetas, params):
    pc,pt = params
    t_pose = pt[:3]
    t_quat = pt[3:]
    cam_pose, cam_orien = fk(thetas, pc)
    cam_quat = R.from_euler('xyz', cam_orien, degrees=True).as_quat()
    return np.linalg.norm(t_pose - cam_pose), np.linalg.norm(t_quat - cam_quat), 0, 0, 0

def ik_cost(thetas, params):
    pc, pt = params
    t_pose = pt[:3]
    t_quat = pt[3:]
    cam_pose, cam_orien = fk(thetas, pc)
    cam_quat = R.from_euler('xyz', cam_orien, degrees=True).as_quat()

    # 포즈와 방향의 오차를 비용으로 계산
    position_error = np.linalg.norm(t_pose - cam_pose)
    orientation_error = np.linalg.norm(t_quat - cam_quat)
    return position_error + orientation_error

class VisualServoBot(YouBot):
    def __init__(self):
        super().__init__()
        self.f_len = self.getGlobalFocalLength()
        self.diff_sum = 0
        self.pixelErrorMin = 10000
        self.stop_flag = False
        self.joints  = []
        for i,arm in enumerate(self.arms):
            jPos = SETTING['joint'][i]
            self.sim.setJointPosition(arm, np.radians(jPos))
        # Gripper Joint
        self.sim.setJointPosition(self.gripper, SETTING['gripper'])
        self.joints += self.arms
        self.joints.append(self.gripper)
        self.cam_pose = self.sim.getObjectPosition(self.camera_1)

        ee_position = self.sim.getObjectPosition(self.camera_1)
        ee_angle = self.sim.getObjectOrientation(self.camera_1) # radian
        self.targetPose = ee_position + ee_angle
        js = self.read_joints(self.joints)
        ps = self.read_points(self.targetPose)
        self.target_thetas = self.solve(js, ps)
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
        circle_img, self.ballPixelPositions = self.detect_features(bgr_img)
        if len(self.ballPixelPositions) == len(TARGET):
            pixel_error = np.linalg.norm(TARGET - np.array(self.ballPixelPositions))
            print("pixel error",pixel_error, self.pixelErrorMin)
            if (pixel_error < 100):
                self.stop_flag = True
            if self.pixelErrorMin >= pixel_error:
                self.pixelErrorMin = pixel_error
        
        # print(self.ballPixelPositions)
        # for i,joint in enumerate(self.joints[:5]):
        #     jPos = SETTING['joint'][i]
        #     self.sim.setJointTargetPosition(joint, np.radians(jPos))
        if self.stop_flag:
            pass
        elif self.diff_sum == 0:
            self.target_thetas, self.diff_sum = self.calculateTarget()
        else:
            self.diff_sum = self.trace_joint(self.joints, self.target_thetas)
            print(self.diff_sum)
            if self.diff_sum < 0.005:
                self.diff_sum = 0

    def calculateTarget(self):
        J = self.CalculateJacobian(self.ballPixelPositions)
        ee_position = self.sim.getObjectPosition(self.camera_1)
        ee_angle = self.sim.getObjectOrientation(self.camera_1)
        self.targetPose = ee_position + ee_angle
        if len(J) > 0:
            velPixel = (TARGET - np.array(self.ballPixelPositions)).flatten()
            Jpinv = np.linalg.pinv(J)
            velCam = Jpinv@velPixel
            x, y, z = velCam[:3]/10
            a, b, c = velCam[3:]*1.2
            # print(f"output:{[x,y,z]},{np.round(np.degrees(np.array([a,b,c])),2)}")
            self.targetPose[0] += x
            self.targetPose[1] -= y
            self.targetPose[2] -= z
            self.targetPose[3] += np.clip(a,-0.01, 0.01)
            self.targetPose[4] += np.clip(b,-0.1, 0.1)
            self.targetPose[5] += np.clip(c,-0.1, 0.1)
            for i in range(3,6):
                if self.targetPose[i] >= np.pi:
                    self.targetPose[i] -= 2*np.pi
                elif self.targetPose[i] < -np.pi:
                    self.targetPose[i] += 2*np.pi
            
            dummy_handle = self.sim.createDummy(0.01)  # Size of dummy
            self.sim.setObjectPosition(dummy_handle, -1, self.targetPose[:3])
            self.sim.setObjectAlias(dummy_handle, f"Sample")
            # print("target pos!: ",self.targetPose[:3])
            # print("target angle!: ",np.round(np.degrees(self.targetPose[3:]),2))
            js = self.read_joints(self.joints)
            ps = self.read_points(self.targetPose) # TODO
            target_thetas = self.solve(js, ps)
            print("target_thetas :",np.round(np.degrees(target_thetas),2),SETTING['t_joint'])
            diff_sum = 0
            for i, target in enumerate(target_thetas):
                diff = target - js[i]
                diff_sum += abs(diff)
            return target_thetas, diff_sum
        else:
            js = self.read_joints(self.joints)
            ps = self.read_points(self.targetPose) # TODO
            target_thetas = self.solve(js, ps)
            diff_sum = 0
            for i, target in enumerate(target_thetas):
                diff = target - js[i]
                diff_sum += abs(diff)
            return target_thetas, diff_sum 

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
        points.append(self.sim.getObject(f"/camera_1"))
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
        j0 = js[0]
        initial_thetas = np.array([js[0], js[1], js[2], js[3], js[4]])
        params = [ps[0], pt]
        theta_bounds = [(np.deg2rad(-0.5),np.deg2rad(0.5)),
                        (np.deg2rad(-80),np.deg2rad(-50)),
                        (np.deg2rad(-80),np.deg2rad(-50)),
                        (np.deg2rad(-90),np.deg2rad(-30)),
                        (np.deg2rad(-45),np.deg2rad(45))]

        result = minimize(
            ik_cost,                     # 목적 함수
            initial_thetas,              # 초기값
            args=(params,),              # 추가 매개변수
            bounds=theta_bounds,         # 범위 제한
            method='L-BFGS-B',           # 제약 조건을 지원하는 최적화 알고리즘
            options={'ftol': 1e-9}       # 수렴 기준
        )

        return result.x
    
    def trace_joint(self, joints, target_thetas):
        js = self.read_joints(joints)
        diff_sum = 0
        thetas = []
        for i, target in enumerate(target_thetas):
            diff = target - js[i]
            if diff > 0.005:
                diff = 0.005
            elif diff < -0.005:
                diff = -0.005
            thetas.append(js[i]+diff)
            diff_sum += abs(diff)
        for joint, theta in zip(joints, thetas):
            self.sim.setJointTargetPosition(joint, theta)
        return diff_sum

    def getGlobalFocalLength(self):
        res, perspAngle = self.sim.getObjectFloatParameter(self.camera_1, self.sim.visionfloatparam_perspective_angle)
        res, resolution = self.sim.getVisionSensorResolution(self.camera_1)
        # distance per pixel
        planeWidth = 2 * math.tan(perspAngle / 2)
        # print(planeWidth, resolution)
        distancePerPixel = planeWidth / resolution
        # global focal length
        pixelFocalLength = (resolution / 2) / math.tan(perspAngle / 2)
        globalFocalLength = pixelFocalLength * distancePerPixel
        return 1/distancePerPixel

    # def detect_features(self, image):
    #     # BGR에서 HSV로 변환
    #     image = cv2.flip(image.copy(),1)
    #     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #     ballPixelPositions = []
    #     for color, ranges in COLORS.items():
    #         mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    #         for lower, upper in ranges:
    #             mask += cv2.inRange(hsv, lower, upper)
        
    #         # 노이즈 제거
    #         kernel = np.ones((3,3), np.uint8)
    #         mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
    #         # 윤곽선 찾기
    #         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #         if contours:
    #             for contour in contours:
    #                 M = cv2.moments(contour)
    #                 # 중심 좌표 계산
    #                 if M["m00"] != 0:
    #                     cX = (M["m10"] / M["m00"])
    #                     cY = (M["m01"] / M["m00"])
    #                     # 원 그리기
    #                     ballPixelPositions.append([cX, cY])
    #                     cv2.circle(image, (int(cX), int(cY)), 3, (255, 255, 0), -1)
    #     image = cv2.flip(image,1)
    #     return image, ballPixelPositions

    def detect_features(self, image):
        # BGR에서 HSV로 변환
        image = image.copy()
        ballPixelPositions = []
        # Aruco 사전 및 파라미터 설정
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()

        # 이미지 읽기 (회전된 마커 포함)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Aruco 마커 감지
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
        # ids = np.squeeze(ids)
        # 4. 탐지된 마커 처리
        if ids is not None:
            if len(ids) >= 2:
                ids_array = ids.flatten()
                # argsort()를 사용해 인덱스 얻기
                sorted_indices = np.argsort(ids_array)
                ids = ids[sorted_indices]
                corners = [corners[i] for i in sorted_indices]
                for corner in corners:
                    corner = np.squeeze(corner)
                    for feat in corner:
                        x,y = feat
                        ballPixelPositions.append([x, y])
                        cv2.circle(image, (int(x), int(y)), 3, (255, 255, 0), -1)
        return image, ballPixelPositions
        
    def getImageJacobian(self, z, u, v):
        img_jacobian = np.array([[-self.f_len/z, 0, u/z, (u*v)/self.f_len, -self.f_len - (u**2)/self.f_len, v],
            [0, -self.f_len/z, v/z, self.f_len + (v**2)/self.f_len, -(u*v)/self.f_len, -u]])
        return img_jacobian

    def CalculateJacobian(self, ballPixelPositions):
        x, y, z = self.sim.getObjectPosition(self.camera_1)
        img_jacobians = []
        for position,t_position in zip(ballPixelPositions, TARGET):
            u,v = position
            tu,tv = t_position
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
