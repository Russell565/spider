import cv2
import numpy as np
import math


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    x = x * 180.0 / 3.141592653589793
    y = y * 180.0 / 3.141592653589793
    z = z * 180.0 / 3.141592653589793

    return np.array([x, y, z])


class Eestimation():

    def Estimation(self, landmarks, img):

        size = img.shape
        # estimate internals
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        print("Camera Matrix :\n {0}".format(camera_matrix))

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

        # 3D model points.
        model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corne
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner
        ])

        for i in range(len(landmarks)):
            print(landmarks[i].shape)
            nose_tip = landmarks[i][30, :]
            chin = landmarks[i][8, :]
            lelc = landmarks[i][36, :]
            rerc = landmarks[i][45, :]
            lmc = landmarks[i][48, :]
            rmc = landmarks[i][54, :]
            image_points = np.array([nose_tip,  # 鼻子
                                     chin,  # Chin
                                     lelc,  # Left eye left corner
                                     rerc,  # Right eye right corne
                                     lmc,  # Left Mouth corner
                                     rmc  # Right mouth corner
                                     ], dtype="double")
            print(image_points.shape)
            # rotation_vector：旋转向量； translation_vector:平移向量
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                          dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

            print('旋转向量: ', rotation_vector)
            # 旋转向量转换旋转矩阵。R为旋转矩阵，j为输出雅可比矩阵(输入数组对输出数组的偏导数)
            rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
            print('旋转矩阵: ', rvec_matrix)

            # 旋转矩阵转换欧拉角
            proj_matrix = np.hstack((rvec_matrix, translation_vector))
            eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]  # 把投影矩阵分解成旋转矩阵和相机矩阵

            pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]
            pitch = math.degrees(math.asin(math.sin(pitch)))
            roll = -math.degrees(math.asin(math.sin(roll)))
            yaw = math.degrees(math.asin(math.sin(yaw)))
            euler_angles = (pitch, yaw, roll)

            print('欧拉角: ', euler_angles)

            axis = np.float32([[400, 0, 0],
                               [0, 400, 0],
                               [0, 0, 400]])
            (nose_end_point2D, jacobian) = cv2.projectPoints(axis, rotation_vector,
                                                             translation_vector, camera_matrix, dist_coeffs)
            for p in image_points:
                cv2.circle(img, (int(p[0]), int(p[1])), 1, (0, 0, 255), -1)

            nose = (int(image_points[0][0]), int(image_points[0][1]))

            cv2.line(img, nose, tuple(int(i) for i in nose_end_point2D[0].ravel()), (255, 0,), 2)  # BLUE
            cv2.line(img, nose, tuple(int(i) for i in nose_end_point2D[1].ravel()), (0, 255, 0), 2)  # GREEN
            cv2.line(img, nose, tuple(int(i) for i in nose_end_point2D[2].ravel()), (0, 0, 255), 2)  # RED

            # pitch: 俯仰角，表示物体绕x轴旋转，上下点头
            # yaw:偏航角，表示物体绕y轴旋转，左右摇头
            # roll:翻滚角，表示物体绕z轴旋转，左右摆头
            text = 'pitch:{:.2f}\nyaw:{:.2f}\nroll:{:.2f}'.format(euler_angles[0],
                                                                  euler_angles[1],
                                                                  euler_angles[2])

            point = (10, 10)
            font_face = cv2.FONT_HERSHEY_PLAIN
            font_scale = 0.8
            thickness = 1
            text_line = text.split('\n')
            text_size, baseline = cv2.getTextSize(str(text_line)
                                                  , font_face
                                                  , font_scale
                                                  , thickness)

            for i, text in enumerate(text_line):
                if text:
                    draw_point = [point[0], point[1] + (text_size[1] + 2 + baseline) * i]
                    img = draw_text(img, text, draw_point, font_face, font_scale, thickness)

            not_use = False
            return img, not_use

    def total_img(self, landmarks, img):
        return self.Estimation(landmarks, img)


def draw_text(img, text, point, font_face, font_scale, thickness):
    bg_color = (0, 255, 0)
    text_size, baseline = cv2.getTextSize(str(text), font_face, font_scale, thickness)
    text_loc = (point[0], point[1] + text_size[1])
    # draw score value
    cv2.putText(img, str(text), (text_loc[0], text_loc[1] + baseline), font_face, font_scale,
                bg_color, thickness, 8)

    return img
