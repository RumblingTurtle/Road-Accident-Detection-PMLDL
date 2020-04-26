import pathlib
import copy
import numpy as np
import cv2


class WPDataset:
    def __init__(self, root, transform=None, target_transform=None,
                 dataset_type='train', balance_data=False,
                 class_names=('BACKGROUND', 'pedestrian', 'rider', 'partially-visible person',
                              'ignore region', 'crowd'), limit=100):
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform

        self.dataset_type = dataset_type.lower()
        self.class_names = class_names
        self.limit = limit
        self.data, self.class_dict = self._read_data()

        self.balance_data = balance_data
        self.ids = [info['image_id'] for info in self.data]
        self.class_stat = None

    def _read_data(self):
        with open(f"{self.root}/{self.dataset_type}.txt", 'r') as fp:
            ids = [line[:-1] for line in fp.readlines()]
        class_dict = {i: class_name for i, class_name in enumerate(self.class_names)}
        data = []
        needed_classes = [1]
        count = 1
        for i in ids:
            if count > self.limit:
                break
            count += 1
            image = i + '.jpg'
            annotation = image + '.txt'
            with open(f"{self.root}/Annotations/{annotation}", 'r') as fp:
                detections = fp.readlines()
            boxes = []
            labels = []
            for j in range(1, len(detections)):
                detection = detections[j].split()
                label = int(detection[0])
                if label not in needed_classes:
                    continue
                labels.append(label)
                bbox = [np.float32(coordinate) for coordinate in detection[1:]]
                boxes.append(bbox)
            if len(boxes) == 0 or len(labels) == 0:
                continue
            boxes = np.array(boxes)
            labels = np.array(labels)
            data.append({
                'image_id': f"{self.root}/Images/{i}.jpg",
                'boxes': boxes,
                'labels': labels
            })
        return data, class_dict

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
        return image, boxes, labels, info

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _read_image(image_id):
        image = cv2.imread(image_id)
        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
