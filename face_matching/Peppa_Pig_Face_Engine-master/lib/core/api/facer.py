import cv2
import numpy as np
import time

from lib.core.api.face_landmark import FaceLandmark
from lib.core.api.face_detector import FaceDetector
from lib.core.LK.lk import GroupTrack,OneEuroFilter,EmaFilter

from config import config as cfg


def find_pupil(landmarks, Xt_raw):
    xmax = int(landmarks[:, 0].max())
    xmin = int(landmarks[:, 0].min())
    ymax = int(landmarks[:, 1].max())
    ymin = int(landmarks[:, 1].min())
    eye_img_bgr = Xt_raw[ymin:ymax, xmin:xmax, :]
    eye_img = cv2.cvtColor(eye_img_bgr, cv2.COLOR_BGR2GRAY)
    eye_img = cv2.equalizeHist(eye_img)
    n_marks = landmarks - np.array([xmin, ymin]).reshape([1, 2])
    eye_mask = cv2.fillConvexPoly(np.zeros_like(eye_img), n_marks.astype(np.int32), 1)
    ret, thresh = cv2.threshold(eye_img, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh = (1-thresh/255.)*eye_mask
    cnt = 0
    xm = []
    ym = []
    for i in range(thresh.shape[0]):
        for j in range(thresh.shape[1]):
            if thresh[i, j] > 0.5:
                xm.append(j)
                ym.append(i)
                cnt += 1
    if cnt != 0:
        xm.sort()
        ym.sort()
        xm = xm[cnt//2]
        ym = ym[cnt//2]
    else:
        xm = thresh.shape[1]/2
        ym = thresh.shape[0]/2
    return xm+xmin, ym+ymin


def get_f5p_from_68p(landmarks, Xt_raw):
    # eye_left = find_pupil(landmarks[36:41], Xt_raw)
    # eye_right = find_pupil(landmarks[42:47], Xt_raw)
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


class FaceAna():

    '''
    by default the top3 facea sorted by area will be calculated for time reason
    '''
    def __init__(self, root_path):

        
        self.face_detector = FaceDetector(root_path)
        self.face_landmark = FaceLandmark(root_path)
        self.trace = GroupTrack()



        ###another thread should run detector in a slow way and update the track_box
        self.track_box=None
        self.previous_image=None
        self.previous_box=None

        self.diff_thres=5
        self.top_k = cfg.DETECT.topk
        self.min_face=cfg.DETECT.min_face
        self.iou_thres=cfg.TRACE.iou_thres
        self.alpha=cfg.TRACE.smooth_box

        if 'ema' in cfg.TRACE.ema_or_one_euro:
            self.filter = EmaFilter(self.alpha)
        else:
            self.filter = OneEuroFilter()

    def run(self,image):
        if True:#self.diff_frames(self.previous_image,image):
            boxes = self.face_detector(image)
            self.previous_image=image
            boxes = self.judge_boxs(self.track_box, boxes)

        else:
            boxes=self.track_box
            self.previous_image = image

        boxes=self.sort_and_filter(boxes)
        boxes_return = np.array(boxes)

        landmarks,states=self.face_landmark.batch_call(image,boxes)

        landmarks = self.trace.calculate(image, landmarks)

        track=[]
        for i in range(landmarks.shape[0]):
            track.append([np.min(landmarks[i][:,0]),np.min(landmarks[i][:,1]),np.max(landmarks[i][:,0]),np.max(landmarks[i][:,1])])
        tmp_box=np.array(track)

        self.track_box = self.judge_boxs(boxes_return, tmp_box)

        return self.track_box,landmarks,states

    def get_f5p(self, image):
        if self.diff_frames(self.previous_image,image):
            boxes = self.face_detector(image)
            self.previous_image=image
            boxes = self.judge_boxs(self.track_box, boxes)

        else:
            boxes=self.track_box
            self.previous_image = image

        boxes=self.sort_and_filter(boxes)
        boxes_return = np.array(boxes)

        landmarks,states=self.face_landmark.batch_call(image,boxes)

        landmarks = self.trace.calculate(image, landmarks)

        track=[]
        for i in range(landmarks.shape[0]):
            track.append([np.min(landmarks[i][:,0]),np.min(landmarks[i][:,1]),np.max(landmarks[i][:,0]),np.max(landmarks[i][:,1])])
        tmp_box=np.array(track)

        self.track_box = self.judge_boxs(boxes_return, tmp_box)
        f5p = [get_f5p_from_68p(lmk, image) for lmk in landmarks]

        return self.track_box, f5p, states

    def diff_frames(self,previous_frame,image):
        '''
        diff value for two value,
        determin if to excute the detection

        :param previous_frame:  RGB  array
        :param image:           RGB  array
        :return:                True or False
        '''
        if previous_frame is None:
            return True
        elif previous_frame.shape != image.shape:
            return True
        else:

            _diff = cv2.absdiff(previous_frame, image)

            diff=np.sum(_diff)/previous_frame.shape[0]/previous_frame.shape[1]/3.

            if diff>self.diff_thres:
                return True
            else:
                return False

    def sort_and_filter(self,bboxes):
        '''
        find the top_k max bboxes, and filter the small face

        :param bboxes:
        :return:
        '''

        if len(bboxes)<1:
            return []


        area=(bboxes[:,2]-bboxes[:,0])*(bboxes[:,3]-bboxes[:,1])
        select_index=area>self.min_face

        area=area[select_index]
        bboxes=bboxes[select_index,:]
        if bboxes.shape[0]>self.top_k:
            picked=area.argsort()[-self.top_k:][::-1]
            sorted_bboxes=[bboxes[x] for x in picked]
        else:
            sorted_bboxes=bboxes
        return np.array(sorted_bboxes)

    def judge_boxs(self,previuous_bboxs,now_bboxs):
        '''
        function used to calculate the tracking bboxes

        :param previuous_bboxs:[[x1,y1,x2,y2],... ]
        :param now_bboxs: [[x1,y1,x2,y2],... ]
        :return:
        '''
        def iou(rec1, rec2):


            # computing area of each rectangles
            S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
            S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

            # computing the sum_area
            sum_area = S_rec1 + S_rec2

            # find the each edge of intersect rectangle
            x1 = max(rec1[0], rec2[0])
            y1 = max(rec1[1], rec2[1])
            x2 = min(rec1[2], rec2[2])
            y2 = min(rec1[3], rec2[3])

            # judge if there is an intersect
            intersect =max(0,x2-x1) * max(0,y2-y1)

            return intersect / (sum_area - intersect)

        if previuous_bboxs is None:
            return now_bboxs

        result=[]

        for i in range(now_bboxs.shape[0]):
            contain = False
            for j in range(previuous_bboxs.shape[0]):
                if iou(now_bboxs[i], previuous_bboxs[j]) > self.iou_thres:
                    result.append(self.smooth(now_bboxs[i],previuous_bboxs[j]))
                    contain=True
                    break
            if not contain:
                result.append(now_bboxs[i][0:4])


        return np.array(result)

    def smooth(self,now_box,previous_box):

        return self.filter(now_box[:4], previous_box[:4])






    def reset(self):
        '''
        reset the previous info used foe tracking,

        :return:
        '''
        del self.track_box
        del self.previous_box
        del self.previous_image
        self.track_box = None
        self.previous_image = None
        self.previous_box = None


