import os
from lib.core.api.facer import FaceAna
from mtcnn_pytorch.src.align_trans import *
from matplotlib import pyplot as plt
from PIL import Image
import random

facer = FaceAna('.')


def splitData(img_path, shuffle=False, ratio=0.9):

    print('切分数据集')
    img_dirs = os.listdir(img_path)  # 文件夹
    # 循环每个人物名称文件夹
    image_list = os.listdir(img_path)
    image_list = [x for x in image_list if 'jpg' in x or 'png' in x]
    image_list.sort()

    offset = int(len(image_list) * ratio)
    if len(image_list) == 0 or offset < 1:
        return [], []
    if offset == len(image_list):
        return image_list, image_list
    if shuffle:
        random.shuffle(image_list)

    train_list = image_list[:offset]
    test_list = image_list[offset:]

    return train_list, test_list


def get_key_point(img_url, data_list, face_list, kp_num=17):

    print('检测人脸关键点')
    img_dirs = os.listdir(img_path)  # 文件夹

    not_det_List = []
    rep_det_List = []
    url_list = []
    landmarks_list = []

    # 循环每个图片
    for image_name in data_list:

        image = cv2.imread(os.path.join(img_url, image_name))

        # pattern = np.zeros_like(image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes, landmarks, states = facer.run(image)
        facer.reset()

        img_show = image.copy()
        # for face_index in range(landmarks.shape[0]):
        #
        #     for landmarks_index in range(landmarks[face_index].shape[0]):
        #         x_y = landmarks[face_index][landmarks_index]
        #         if landmarks_index > 16:
        #             cv2.circle(img_show, (int(x_y[0]), int(x_y[1])), 3,
        #                        (0, 0, 255), -1)
        #             landmarks_list.append(x_y)
        #
        #         else:
        #             cv2.circle(img_show, (int(x_y[0]), int(x_y[1])), 3,
        #                        (222, 222, 222), -1)
        #             landmarks_list.append(x_y)
        #
        # cv2.namedWindow("capture", 0)
        # cv2.imshow("capture", img_show)

        # print(landmarks_list)
        # print(Xses)
        # if args.mask:
        #     cv2.namedWindow("masked", 0)
        #     cv2.imshow("masked", image * pattern)

        if landmarks.shape[0] > 0:

            align_list = align(landmarks, image)
            boxes, landmarks, states = facer.run(align_list)
            if landmarks.shape[0] < 1:
                print('这张没有检测到人脸: ', os.path.join(img_url, image_name))
                not_det_List.append(os.path.join(img_url, image_name))
                break
            if len(landmarks.flatten()) > 136:
                print('这张检测到多人脸: ', os.path.join(img_url, image_name))
                rep_det_List.append(os.path.join(img_url, image_name))
                break

            # 17个轮廓点
            if kp_num == 17:
                new_landmarks = del_f5p(landmarks[0])
                face_list.append(new_landmarks.flatten())  # 17p
            else:
                face_list.append(landmarks.flatten())  # 68p
            url_list.append(os.path.join(img_url, image_name))

            # for face_index in range(new_landmarks.shape[0]):
            #
            #     for landmarks_index in range(new_landmarks[face_index].shape[0]):
            #         x_y = new_landmarks[face_index][landmarks_index]
            #         cv2.circle(align_list, (int(x_y[0]), int(x_y[1])), 3,
            #                    (222, 222, 222), -1)
            #
            # cv2.namedWindow("capture2", 0)
            # cv2.imshow("capture2", align_list)
        else:
            print('这张没有检测到人脸: ', os.path.join(img_url, image_name))
            not_det_List.append(os.path.join(img_url, image_name))

        # key = cv2.waitKey(0)
        # if key == ord('q'):
        #     return
        # if len(face_list) > 1500:  # 先使用1000张图片
        #     break

    print('没检测到', len(not_det_List))
    print(not_det_List)
    print('多人脸检测到', len(rep_det_List))
    print(rep_det_List)
    return not_det_List, url_list, rep_det_List


def align(landmarks_list, image):

    f5p = get_f5p(landmarks_list[0])
    align_face = warp_and_crop_face(image, f5p
                                    , reference_pts=get_reference_facial_points(default_square=True)
                                    , crop_size=(256, 256))

    # cv2.namedWindow("align_face", 0)
    # cv2.imshow("align_face", align_face)

    return align_face


def do_pca(data, k):

    data = np.float32(np.mat(data))
    # 取大小
    rows, cols = data.shape
    # 求每列均值
    data_mean = np.mean(data, 0)
    # 中心化，每行减去均值。
    Z = data - np.tile(data_mean, (rows, 1))
    # Z.T*Z 矩阵转方阵。获取特征值与特征向量
    eig_vals, eig_vects = np.linalg.eig(np.matmul(Z.T, Z))

    # argsort():对特征值矩阵进行由小到大排序，返回对应排序后的索引
    eig_val_ind = np.argsort(eig_vals)
    del eig_vals
    # 从排序后的矩阵最后一个开始自下而上选取最大的N个特征值，返回其对应的索引
    eig_val_ind = eig_val_ind[: -(k+1): -1]
    # 将特征值最大的N个特征值对应索引的特征向量提取出来，组成压缩矩阵
    new_eig_vects = eig_vects[:, eig_val_ind]
    del eig_vects
    # 特征向量归一化
    for i in range(k):
        new_eig_vects[:, i] /= np.linalg.norm(new_eig_vects[:, i])
    # 将去除均值后的数据矩阵*压缩矩阵，转换到新的空间，使维度降低为k
    new_data = np.matmul(Z, new_eig_vects)

    return np.array(new_data), data_mean, new_eig_vects


def get_distance(test_list, face_list):

    result = []
    for i in range(len(face_list)):
        dist = eucli_dist(test_list, face_list[i])
        result.append(dist)

    return result


def eucli_dist(list_x, list_y):

    return np.sqrt(sum(np.power((list_x - list_y), 2)))


def del_f5p(landmarks):
    landmarks = landmarks.tolist()

    del landmarks[17:]
    result = []
    result.append(landmarks)

    return np.array(result)


def get_f5p(landmarks):
    eye_left = landmarks[36:41].mean(axis=0)
    eye_right = landmarks[42:47].mean(axis=0)
    nose = landmarks[30]
    mouth_left = landmarks[48]
    mouth_right = landmarks[54]
    f5p = [[eye_left[0], eye_left[1]],
           [eye_right[0], eye_right[1]],
           [nose[0], nose[1]],
           [mouth_left[0], mouth_left[1]],
           [mouth_right[0], mouth_right[1]]]
    return f5p


if __name__ == '__main__':

    # img_path = '/home/wwgz-cbm/img/test'
    img_path = './test_img/img'
    ratio = 0.8
    # 切分数据集
    train_data_list, test_data_list = splitData(img_path, shuffle=True, ratio=ratio)

    face_list_kp17 = []
    face_list_kp68 = []
    # 获取人脸关键点、对齐
    _, kp17_url_list, _ = get_key_point(img_path, train_data_list, face_list_kp17, kp_num=17)
    _, kp68_url_list, _ = get_key_point(img_path, train_data_list, face_list_kp68, kp_num=68)

    # PCA降维，获取前95%特征
    face_list_kp17, data_mean_kp17, V_kp17 = do_pca(face_list_kp17, int(len(face_list_kp17[0])*0.95))
    face_list_kp68, data_mean_kp68, V_kp68 = do_pca(face_list_kp68, int(len(face_list_kp68[0])*0.95))

    # 测试数据
    test_list_kp17 = []
    test_list_kp68 = []
    _, test_url_list_kp17, _ = get_key_point(img_path, test_data_list, test_list_kp17, kp_num=17)
    _, test_url_list_kp68, _ = get_key_point(img_path, test_data_list, test_list_kp68, kp_num=68)

    num_test = len(test_list_kp17)
    temp_face_kp17 = test_list_kp17 - np.tile(data_mean_kp17, (num_test, 1))
    temp_face_kp68 = test_list_kp68 - np.tile(data_mean_kp68, (num_test, 1))
    data_test_new_kp17 = temp_face_kp17 * V_kp17
    data_test_new_kp68 = temp_face_kp68 * V_kp68
    data_test_new_kp17 = np.array(data_test_new_kp17)
    data_test_new_kp68 = np.array(data_test_new_kp68)

    result_kp17 = []
    result_kp68 = []
    for i in range(num_test):
        plt.figure()

        test_img = Image.open(test_url_list_kp17[i])
        plt.subplot(2, 2, 1)
        plt.title('test_img_kp17: ' + test_url_list_kp17[i][len(img_path):])
        plt.imshow(test_img)
        plt.axis('off')

        test_img = Image.open(test_url_list_kp68[i])
        plt.subplot(2, 2, 3)
        plt.title('test_img_kp68: ' + test_url_list_kp68[i][len(img_path):])
        plt.imshow(test_img)
        plt.axis('off')

        test_face = data_test_new_kp17[i, :]
        result_kp17.append(get_distance(test_face, face_list_kp17))  # 计算欧式距离

        tran_img = Image.open(kp17_url_list[result_kp17[i].index(min(result_kp17[i]))])
        plt.subplot(2, 2, 2)
        plt.title('train_img_kp17: ' + kp17_url_list[result_kp17[i].index(min(result_kp17[i]))][len(img_path):])
        plt.imshow(tran_img)
        plt.axis('off')

        test_face = data_test_new_kp68[i, :]
        result_kp68.append(get_distance(test_face, face_list_kp68))  # 计算欧式距离

        tran_img = Image.open(kp68_url_list[result_kp68[i].index(min(result_kp68[i]))])
        plt.subplot(2, 2, 4)
        plt.title('train_img_kp68: ' + kp68_url_list[result_kp68[i].index(min(result_kp68[i]))][len(img_path):])
        plt.imshow(tran_img)
        plt.axis('off')

        if not os.path.exists('/home/wwgz-cbm/Pictures/' + str(ratio) + 'ratio/'):
            os.makedirs('/home/wwgz-cbm/Pictures/' + str(ratio) + 'ratio/')
        plt.savefig('/home/wwgz-cbm/Pictures/' + str(ratio) + 'ratio/' + str(i+1))
        plt.show()

        print('测试图片%s: kp17距离 %f; kp68距离 %f' % (test_url_list_kp17[i], min(result_kp17[i]), min(result_kp68[i])))
        print('kp17匹配结果: %s; kp17匹配结果: %s' % (kp17_url_list[result_kp17[i].index(min(result_kp17[i]))],
                                              kp68_url_list[result_kp68[i].index(min(result_kp68[i]))]))
