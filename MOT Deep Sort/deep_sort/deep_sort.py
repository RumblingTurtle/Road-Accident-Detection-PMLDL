import numpy as np
import torch

from .deep.feature_extractor import Extractor
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.preprocessing import non_max_suppression
from .sort.detection import Detection
from .sort.tracker import Tracker

__all__ = ['DeepSort']
NORMAL = 0
MOVE = 1
COLLISION = 2


class DeepSort(object):
    def __init__(self, model_path, max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0, max_iou_distance=0.7,
                 max_age=70, n_init=3, nn_budget=100, use_cuda=True):
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap

        self.extractor = Extractor(model_path, use_cuda=use_cuda)

        max_cosine_distance = max_dist
        nn_budget = 100
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    def update(self, bbox_xywh, confidences, ori_img):
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        features = self._get_features(bbox_xywh, ori_img)
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(confidences) if
                      conf > self.min_confidence]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            collision_status = track.collision_status
            outputs.append(np.array([x1, y1, x2, y2, collision_status, track_id], dtype=np.int))
            if len(track.acc_q) > 2:
                self.detect_collision(track)
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 2] = bbox_xywh[:, 2] - bbox_xywh[:, 0]
        bbox_tlwh[:, 3] = bbox_xywh[:, 3] - bbox_xywh[:, 1]
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x + w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), self.height - 1)
        return x1, y1, x2, y2

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features

    @staticmethod
    def intersect(box_a, box_b):
        x_overlap = box_a[2] >= box_b[0] and box_b[2] >= box_a[0]
        y_overlap = box_a[3] >= box_b[1] and box_b[3] >= box_a[1]
        return x_overlap and y_overlap

    def detect_collision(self, track):
        vel_q = track.vel_q
        acc_q = track.acc_q
        sign_x = 1 if (vel_q[0][0] > 0 - vel_q[0][0] < 0) == (acc_q[0][0] > 0 - acc_q[0][0] < 0) else -1
        sign_y = 1 if (vel_q[0][1] > 0 - vel_q[0][1] < 0) == (acc_q[0][1] > 0 - acc_q[0][1] < 0) else -1
        avg_acc = [0, 0]
        if len(acc_q) > 1:
            lim = min(len(acc_q), 3)
            for i in range(1, lim):
                sign_x_i = 1 if (vel_q[i][0] > 0 - vel_q[i][0] < 0) == (acc_q[i][0] > 0 - acc_q[i][0] < 0) else -1
                sign_y_i = 1 if (vel_q[i][1] > 0 - vel_q[i][1] < 0) == (acc_q[i][1] > 0 - acc_q[i][1] < 0) else -1
                avg_acc[0] += sign_x_i * abs(acc_q[i][0])
                avg_acc[1] += sign_y_i * abs(acc_q[i][1])
            avg_acc[0] /= lim - 1
            avg_acc[1] /= lim - 1
        threshold = (abs(sign_x * abs(track.acc[0] - avg_acc[0])), abs(sign_y * (track.acc[1] - avg_acc[1])))
        if threshold[0] > 4 or threshold[1] >= 3:
            track.collision_status = MOVE
            for t in self.tracker.tracks:
                if track.track_id == t.track_id:
                    continue
                if self.intersect(track.to_tlbr(), t.to_tlbr()):
                    print(f'Collision between {track.track_id} and {t.track_id}')
                    if t.collision_status == MOVE:
                        track.collision_status = COLLISION
                        t.collision_status = COLLISION
        else:
            track.collision_status = NORMAL
