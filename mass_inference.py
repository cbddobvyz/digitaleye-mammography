import os
import cv2
import glob
import argparse
import numpy as np
from utils.all_utils import *
import pickle
import warnings
import time
import pathlib


# git ignore dosyasi olusturup results komple ignore, 
start_time = time.time()
print('PROCESSES STARTED')
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--model_enum', nargs='+', help='Example: --model_enum 0 or 0 1 2, Purpose: 0: ATSS, 1: Cascade R-CNN, 2: DEFORMABLE DETR, 3: DETR, 4: DOUBLE HEAD R-CNN, 5: DYNAMIC R-CNN, 6: FASTER R-CNN, 7: FCOS, 8: RETINA NET, 9: VARIFOCAL NET, 10: YOLOv3', required=True)
parser.add_argument('--device', type=str, help='Example: --device cuda:0', required=False, default='cpu')
parser.add_argument('--classify_mass', type=str, help='Example: --classify_mass True or False, Purpose: It is for doing classificiation', required=False, default='True') 
parser.add_argument('--segment_breast', type=str , help='Example: --segment_breast True or False, Purpose: It is for applying breast segmentation model', required=False, default='True') 
parser.add_argument('--enable_ensemble', type=str, help='Example: --enable_ensemble True or False, Purpose: It is for applying ensemble strategy to detections', required=False, default='False')
parser.add_argument('--img_path', type=str, help='Example: --PATH, Purpose: It is for getting image folder path', required=True, default=None)
parser.add_argument('--annotation_path', type=str, help='Example: --PATH, Purpose: It is for getting annotation .txt file path', required=False, default=None)
parser.add_argument('--nms_iou_threshold', type=float, help='Example: --nms_iou_threshold 0.1', required=False, default=0.1)
parser.add_argument('--confidence_threshold', type=float, help='Example: --confidence_threshold 0.2', required=False, default=0.2)
parser.add_argument('--ap_threshold', type=float, help='Example: --ap_threshold 0.1', required=False, default=0.1)

args = parser.parse_args()

if len(glob.glob(os.path.join(args.img_path, '*.png'))) == 0:
    print(args.img_path, 'not include any images... You must give image folder which contains images...')
    parser.print_help()
    exit(-1)

if len(args.model_enum) == 1 and args.enable_ensemble == 'True':
    print('You must give more than one model for applying ensemble strategy')
    parser.print_help()
    exit(-1)

model_names = args.model_enum
device = args.device
if args.classify_mass == 'True':
    label_names = ['Malign','Benign']
else:
    label_names = ['Mass']
class_size = len(label_names)
current_dir = os.path.dirname(os.path.realpath(sys.argv[0]))

results_dir = create_result_dir(current_dir)

config_paths, model_file_paths, model_result_paths, selected_model_names = get_choosen_models(current_dir, results_dir, model_names)

print(results_dir, 'directory created...')

if args.segment_breast =='True':
    seg_img_path = os.path.join(results_dir, 'breast_segmentation')
    print('Image files feeding to segmentation model and segmentation results will be saved on', seg_img_path, 'directory.'),
    crop_coordinates, img_shapes = apply_segmentation(seg_img_path, args.img_path, device)
    img_list, ann_list = control_annot_path(args.annotation_path, seg_img_path)
    if args.annotation_path:
        annot_path = get_annot_path(sorted(img_list), ann_list, img_shapes, crop_coordinates, seg_img_path)
        img_list, ann_list = control_annot_path(annot_path, seg_img_path)
    else:
        annot_path = args.annotation_path

else:
    crop_coordinates = None
    annot_path = args.annotation_path
    img_list, ann_list = control_annot_path(args.annotation_path, args.img_path)
    
if ann_list:
    bb_box, _ = get_gtbbox(ann_list)
    annot_classes = []
    for bb in bb_box:
        for b in bb:
            annot_classes.append(int(b[-1]))
    annot_classes = sorted(list(set(annot_classes)))
    if len(label_names) != len(annot_classes):
        print('According to class size, You must prepare annotation path or set --classu must give more than one model for applying ensemble stification parameter. You give'
             , len(annot_classes), 'classes in annotation file and you set --classification parameter as', args.classification, 'it causes not having same size of annotation and label classes. Label classes length and annotation classes length must be same.')
        parser.print_help()
        exit(-1)

write_labels_to_txt(label_names)    

model_predicts = []
df_dict = {}
for i in range(len(config_paths)): 
    print('*'*20, selected_model_names[i], 'model evaluation processes are starting...', '*'*20)
    results = get_model_predicts(config_paths[i], model_file_paths[i], img_list, class_size, device)
    results = get_nms_results(results, img_list, class_size, args.nms_iou_threshold, scr_thr=args.confidence_threshold)
    if class_size == 2:
        results = filter_results(results)
    if ann_list:
    	model_evals(config_paths[i], results, args.ap_threshold, args.img_path, annot_path, class_size)
    save_results(img_list, ann_list, results, label_names, model_result_paths[i])
    
    if args.enable_ensemble == 'True':
        model_predicts.append([results])

        
if len(model_predicts)!=0:
    print('*'*20, 'Applying Ensemble with:', ' '.join(selected_model_names), '*'*20)
    print('*'*20,'ENSEMBLE evaluation processes are starting...', '*'*20)
    
    ensemble_result_path = os.path.join(results_dir, '_'.join(model_names) + '_ensemble')
    print(ensemble_result_path)
    if not os.path.exists(ensemble_result_path):
        os.makedirs(ensemble_result_path)
    ensemble_result = applying_ensemble(img_list, model_predicts, class_size, args.nms_iou_threshold, args.confidence_threshold)
    if class_size == 2:
        ensemble_result = filter_results(ensemble_result)
    if ann_list:
    	model_evals(config_paths[0], ensemble_result, args.ap_threshold, args.img_path, annot_path, class_size)
    save_results(img_list, ann_list, ensemble_result, label_names, ensemble_result_path)
    selected_model_names.append('ENSEMBLE')

end_time = time.time()

print('-*- ELAPSED PROCESSING TIME:', int(end_time-start_time), 'seconds -*-')
