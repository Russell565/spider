import os
from lib.core.api.facer import FaceAna
from mtcnn_pytorch.src.align_trans import *
from face_eastimation import Eestimation

facer = FaceAna('.')


def splitData(img_path):

    image_list = os.listdir(img_path)
    image_list = [x for x in image_list if 'jpg' in x or 'png' in x]
    image_list.sort()

    return image_list


def get_key_point(img_url, data_list):

    not_det_List = []
    rep_det_List = []

    # 循环每个图片
    for image_name in data_list:

        image = cv2.imread(os.path.join(img_url, image_name))

        boxes, landmarks, states = facer.run(image)
        facer.reset()

        if landmarks.shape[0] > 0:

            # 人脸姿态估计
            est = Eestimation()
            est_img, not_use = est.total_img(landmarks, image)
            if not_use:
                break
            filename = '/home/wwgz-cbm/Pictures/est_result/' + image_name
            cv2.imwrite(filename, est_img)
        else:
            print('这张没有检测到人脸: ', os.path.join(img_url, image_name))
            not_det_List.append(os.path.join(img_url, image_name))

    print('没检测到', len(not_det_List))
    print(not_det_List)
    print('多人脸检测到', len(rep_det_List))
    print(rep_det_List)
    return not_det_List, rep_det_List


if __name__ == '__main__':

    img_path = './test_img/angle_img'
    # 切分数据集
    train_data_list, test_data_list = splitData(img_path)

    # 获取人脸关键点、人脸角度估计
    get_key_point(img_path, train_data_list)
