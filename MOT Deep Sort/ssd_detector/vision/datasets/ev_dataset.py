import pathlib
import json
import os
import copy
import numpy as np
import cv2


class EVDataset:
    def __init__(self, root, transform=None, target_transform=None,
                 dataset_type='train', balance_data=False,
                 class_names=('BACKGROUND', 'pedestrian', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6')):
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform

        self.dataset_type = dataset_type.lower()
        self.class_names = class_names
        self.data, self.class_dict = self._read_data()

        self.balance_data = balance_data
        self.ids = [info['image_id'] for info in self.data]
        self.class_stat = None

    def _read_data(self):
        annotations = [file for file in os.listdir(self.root / self.dataset_type)
                       if file.endswith('.json')]
        class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}
        data = []
        for annotation in annotations:
            image_id = annotation[:-len('.json')]
            boxes = []
            labels = []
            with open(f"{self.root}/{self.dataset_type}/{annotation}", 'r') as json_file:
                objects = json.load(json_file)
                for detection in objects:
                    label = detection['class']
                    if label not in self.class_names:
                        continue
                    polygon = detection['polygon']
                    if len(polygon) < 4:
                        continue
                    labels.append(class_dict[label])
                    xs, ys = [np.float32(coordinate[0]) for coordinate in polygon], \
                             [np.float32(coordinate[1]) for coordinate in polygon]
                    boxes.append([min(xs), min(ys), max(xs), max(ys)])
            if len(boxes) == 0 or len(labels) == 0:
                continue
            boxes = np.array(boxes)
            labels = np.array(labels)
            data.append({
                'image_id': str(self.root / self.dataset_type / f"{image_id}.jpg"),
                'boxes': boxes,
                'labels': labels
            })
        return data, class_dict

    def merge(self, dataset):
        self.ids.extend(dataset.ids)
        for d in dataset.data:
            self.data.append(d)
        self.class_stat = None

    def __repr__(self):
        if self.class_stat is None:
            self.class_stat = {name: 0 for name in self.class_names}
            for example in self.data:
                for class_index in example['labels']:
                    class_name = self.class_names[class_index]
                    self.class_stat[class_name] += 1
        content = ["Dataset Summary:",
                   f"Number of Images: {len(self.data)}",
                   "Label Distribution:"]
        for class_name, num in self.class_stat.items():
            content.append(f"\t{class_name}: {num}")
        return "\n".join(content)

    def _getitem(self, index):
        image_info = self.data[index]
        image = self._read_image(image_info['image_id'])
        boxes = copy.copy(image_info['boxes'])
        labels = copy.copy(image_info['labels'])
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image_info['image_id'], image, boxes, labels

    def __getitem__(self, index):
        info, image, boxes, labels = self._getitem(index)
        return image, boxes, labels

    def __len__(self):
        return len(self.data)

    def get_annotation(self, index):
        """To conform the eval_ssd implementation that is based on the EV dataset."""
        image_id, image, boxes, labels = self._getitem(index)
        is_difficult = np.zeros(boxes.shape[0], dtype=np.uint8)
        return image_id, (boxes, labels, is_difficult)

    def get_image(self, index):
        image_info = self.data[index]
        image = self._read_image(image_info['image_id'])
        if self.transform:
            image, _ = self.transform(image)
        return image

    @staticmethod
    def _read_image(image_id):
        image = cv2.imread(image_id)
        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
