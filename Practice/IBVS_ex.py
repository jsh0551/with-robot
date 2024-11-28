import time
import numpy as np
import math
import cv2
from pynput import keyboard
from ModelBase import Base

# TARGET = np.array([[448.65582344686817, 62.34417655313177], [62.34417655313177, 448.65582344686817], [62.395300906842536, 255.5],
#                     [62.34417655313177, 62.34417655313177], [448.65582344686817, 448.65582344686817],
#                     [255.5, 255.5], [371.38202571882675, 294.1710735442855], [255.5, 62.395300906842536]
#                     ])
# COLORS = {
#     'red': [
#         (np.array([0, 100, 100]), np.array([10, 255, 255]))
#         # (np.array([160, 100, 100]), np.array([180, 255, 255]))
#     ],
#     'orange': [
#         (np.array([11, 100, 100]), np.array([25, 255, 255]))
#     ],
#     'yellow': [
#         (np.array([26, 100, 100]), np.array([35, 255, 255]))
#     ],
#     'green': [
#         (np.array([36, 100, 100]), np.array([80, 255, 255]))
#     ],
#     'blue': [
#         (np.array([100, 100, 100]), np.array([130, 255, 255]))
#     ],
#     'purple': [
#         (np.array([131, 100, 100]), np.array([155, 255, 255]))
#     ],
#     'pink': [
#         (np.array([156, 100, 100]), np.array([165, 255, 255]))
#     ],
#     'cyan': [
#         (np.array([81, 100, 100]), np.array([99, 255, 255]))
#     ]
# }

TARGET = np.array([[124.0, 202.53735632183907], [92.8230890464933, 86.91725768321513], [87.7598797250859, 136.79725085910653], [200.8926619828259, 85.00468384074941], [198.43244506778865, 188.51706404862082], [157.65282865282865, 107.45706145706146]])
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

class IBVS(Base):
    def __init__(self):
        super().__init__()
        # coppelia sim init
        self.camera = self.sim.getObject('/camera_1')
        # self.sim.setObjectPosition(self.camera, [0.4, 0.3, 1.0])
        # self.sim.setObjectOrientation(self.camera, [np.radians(-175), np.radians(-15), np.radians(-165)])
        cam_position = self.sim.getObjectPosition(self.camera)
        cam_angle = self.sim.getObjectOrientation(self.camera)
        # print(cam_angle)
        self.cameraPose = cam_position + cam_angle
        print(self.cameraPose)
        self.f_len = self.getGlobalFocalLength()
        print(self.f_len)
        self.ballPixelPositions = []
        
    def actuation(self):
        pass
        # print(self.cam_pos)
        # self.cam_pos[2] += 0.005
        J = self.CalculateJacobian(self.ballPixelPositions)
        print("ball position ",self.ballPixelPositions)
        if len(self.ballPixelPositions) > 0:
            pixel_error = np.linalg.norm(TARGET - np.array(self.ballPixelPositions))
            print("pixel error",pixel_error)
        if len(J) > 0:
            velPixel = (TARGET - np.array(self.ballPixelPositions)).flatten()
        #     print("a",TARGET)
        #     print("b",self.ballPixelPositions)
            Jpinv = np.linalg.pinv(J)
            velCam = Jpinv@velPixel
            x, y, z = velCam[:3]/50
            a, b, c = velCam[3:]/25
            # print(velCam[:3],np.degrees(velCam[3:]/10))
            # print("..",self.cameraPose[:3],np.degrees(self.cameraPose[3:]))
            print(f"output:{[x,y,z]},{np.round(np.degrees(np.array([a,b,c])),2)}")
            velCam = velCam.reshape(-1,2)
            self.cameraPose[0] += x
            self.cameraPose[1] -= y
            self.cameraPose[2] -= z
            self.cameraPose[3] += a
            self.cameraPose[4] += b
            self.cameraPose[5] += c
            dummy_handle = self.sim.createDummy(0.005)  # Size of dummy
            self.sim.setObjectPosition(dummy_handle, -1, self.cameraPose[:3])
            self.sim.setObjectAlias(dummy_handle, f"Sample")
            for i in range(3,6):
                if self.cameraPose[i] > np.pi:
                    self.cameraPose[i] -= 2*np.pi
                elif self.cameraPose[i] < -np.pi:
                    self.cameraPose[i] += 2*np.pi
        #     print(velCam[:3])
        #     print(self.ballPixelPositions)
            print("pos",self.cameraPose[:3])
        #     print("angle",np.degrees(self.cameraPose[3:]))
        self.sim.setObjectPosition(self.camera, self.cameraPose[:3])
        self.sim.setObjectOrientation(self.camera, self.cameraPose[3:])

    def sensing(self):
        img = self.read_camera()
        bgr_img = img[::-1,:,::-1]
        circle_img, self.ballPixelPositions = self.detect_features(bgr_img)
        cv2.imshow("object", circle_img)
        cv2.waitKey(10)
        self.step()

    def read_camera(self):
        result = self.sim.getVisionSensorImg(self.camera)
        img = np.frombuffer(result[0], dtype=np.uint8)
        img = img.reshape((result[1][1], result[1][0], 3))
        return img

    def getGlobalFocalLength(self):
        res, perspAngle = self.sim.getObjectFloatParameter(self.camera, self.sim.visionfloatparam_perspective_angle)
        res, resolution = self.sim.getVisionSensorResolution(self.camera)
        # distance per pixel
        planeWidth = 2 * math.tan(perspAngle / 2)
        # print(planeWidth, resolution)
        distancePerPixel = planeWidth / resolution
        # global focal length
        pixelFocalLength = (resolution / 2) / math.tan(perspAngle / 2)
        globalFocalLength = pixelFocalLength * distancePerPixel
        return 1/distancePerPixel

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
                        # print(color, cX,cY)
                        cv2.circle(image, (int(cX), int(cY)), 3, (255, 255, 0), -1)
        # print("=====")
        return image, ballPixelPositions
        
    def getImageJacobian(self, z, u, v):
        img_jacobian = np.array([[-self.f_len/z, 0, u/z, (u*v)/self.f_len, -self.f_len - (u**2)/self.f_len, v],
            [0, -self.f_len/z, v/z, self.f_len + (v**2)/self.f_len, -(u*v)/self.f_len, -u]])
        return img_jacobian

    def CalculateJacobian(self, ballPixelPositions):
        x, y, z = self.sim.getObjectPosition(self.camera)
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

if __name__ == "__main__":
    ibvs = IBVS()
    ibvs.sim.startSimulation()
    ibvs.set_sync(True)
    while not ibvs.quit:
        try:
            ibvs.actuation()
            ibvs.sensing()
        except Exception as e:
            break
    ibvs.sim.stopSimulation()