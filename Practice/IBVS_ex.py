import time
import numpy as np
import math
import cv2
from pynput import keyboard
from ModelBase import Base

TARGET = np.array([[398.6631286328026, 112.33687136719736], [398.6631286328026, 398.6631286328026], [112.33687136719736, 112.33687136719736]])
COLORS = {
            'red': [
                (np.array([0, 100, 100]), np.array([10, 255, 255])),
                (np.array([160, 100, 100]), np.array([180, 255, 255]))
            ],
            'blue': [(np.array([100, 100, 100]), np.array([130, 255, 255]))],
            'green': [(np.array([40, 100, 100]), np.array([80, 255, 255]))]
        }


class IBVS(Base):
    def __init__(self):
        super().__init__()
        # coppelia sim init
        self.camera = self.sim.getObject('/Camera1')
        cam_position = self.sim.getObjectPosition(self.camera)
        cam_angle = self.sim.getObjectOrientation(self.camera)
        self.cameraPose = cam_position + cam_angle
        self.f_len = self.getGlobalFocalLength()
        self.ballPixelPositions = []
        
    def actuation(self):
        # print(self.cam_pos)
        # self.cam_pos[2] += 0.005
        J = self.CalculateJacobian(self.ballPixelPositions)
        # print(self.ballPixelPositions)
        if len(J) > 0:
            velPixel = -(TARGET - np.array(self.ballPixelPositions)).flatten()
        #     print("a",TARGET)
        #     print("b",self.ballPixelPositions)
            Jpinv = np.linalg.pinv(J)
            velCam = Jpinv@velPixel
            x, y, z = velCam[:3]/5000
            a, b, c = np.degrees(velCam[3:])/100
        #     # velCam = velCam.reshape(-1,2)
            self.cameraPose[0] -= x
            self.cameraPose[1] += y
            self.cameraPose[2] += z*20
            self.cameraPose[3] -= a*20
            self.cameraPose[4] += b*3
            self.cameraPose[5] -= c
            print(velCam[:3])
            print(self.ballPixelPositions)
            print("pos",self.cameraPose[:3])
            print("angle",np.degrees(self.cameraPose[3:]))
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
        ibvs.actuation()
        ibvs.sensing()
    ibvs.sim.stopSimulation()