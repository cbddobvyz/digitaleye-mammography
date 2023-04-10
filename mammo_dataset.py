import mmcv
import numpy as np

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset


import os


@DATASETS.register_module()
class MammoDataset(CustomDataset):
    
    # CLASSES = ('MASS', )
    class_file = open('/workspace/notebooks/new_mammo_repo/classes.txt', 'r')
    class_names = class_file.read()
    CLASSES = tuple(class_names.split(','))
    class_file.close()
    os.remove('/workspace/notebooks/new_mammo_repo/classes.txt')

    def load_annotations(self, ann_file):
        ann_list = mmcv.list_from_file(ann_file)

        data_infos = []
        for i, ann_line in enumerate(ann_list):
            if ann_line != '#':
                continue

            img_shape = ann_list[i + 2].split(' ')
            width = int(img_shape[0])
            height = int(img_shape[1])
            bbox_number = int(ann_list[i + 3])

            
            bboxes = []
            labels = []
            
            for anns in ann_list[i + 4:i + 4 + bbox_number]:
                ann_line = anns.split(' ')
                bboxes.append([float(ann) for ann in ann_line[:4]])
                labels.append(int(ann_line[4]))
                

            data_infos.append(
                dict(
                    filename=ann_list[i + 1],
                    width=width,
                    height=height,
                    ann=dict(
                        bboxes=np.array(bboxes).astype(np.float32),
                        labels=np.array(labels).astype(np.int64))
                ))

        return data_infos


    def get_ann_info(self, idx):
        return self.data_infos[idx]['ann']