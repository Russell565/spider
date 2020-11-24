import os
import face_recognition
from PIL import Image
from PIL import ImageFile
import threading

ImageFile.LOAD_TRUNCATED_IMAGES = True


def process_img(path, new_path):
    pics = os.listdir(path)
    for pic in pics:
        pic_path = os.path.join(path, pic)
        image = face_recognition.load_image_file(pic_path)
        face_locations = face_recognition.face_locations(image)
        if len(face_locations) == 0:
            # os.remove(pic_path)
            # print(f'删除{pic_path}')
            continue
        img = Image.open(pic_path)
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        if len(img.split()) == 4:
            # 利用split和merge将通道从四个转换为三个
            r, g, b, a = img.split()
            toimg = Image.merge("RGB", (r, g, b))
            toimg.save(new_path + '/' + pic)
        else:
            try:
                img.save(new_path + '/' + pic)
            except:
                continue
    print('Finish......!', path)


def lock_test(path, new_path):
    mu = threading.Lock()
    if mu.acquire(True):
        process_img(path, new_path)
        mu.release()


if __name__ == '__main__':
    paths = r'/home/wwgz-cbm/spider_img/testBing'
    new_paths = r'/home/wwgz-cbm/spider_img/img'

    dirs = os.listdir(paths)  # 文件夹
    for name_dir in dirs:

        # my_thread = threading.Thread(target=lock_test
        #                              , args=[paths+'/'+name_dir, new_paths+'/'+name_dir])
        # my_thread.start()
        process_img(paths+'/'+name_dir, new_paths+'/'+name_dir)
        # break
