import sys
import os
from inspect import getsourcefile
from os.path import abspath
sys.path.append(os.path.dirname(os.path.realpath(sys.argv[0])) + '/utils')

import cv2
import glob
import mmcv
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mmcv
from tqdm import tqdm
import numpy as np
import mammoggraphy_preprocess as mam_pre
import warnings

from PIL import Image
from ensemble_boxes import *
from mmcv import Config, DictAction
from mmdet.datasets import build_dataset
from mmdet.apis import inference_detector, init_detector
from multiprocessing import Pool
from mmcv.utils import print_log
from terminaltables import AsciiTable
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.core.evaluation.class_names import get_classes
from mmdet.core.evaluation.mean_ap import *
from sklearn.metrics import auc
from seg_utils import *

warnings.filterwarnings("ignore")

# get_result_array düzelt degiskenleri
def get_choosen_models(current_dir, results_dir, model_names):
    """
    Loading desired models' requirements

    Parameters
    ---------- 
    current_dir : str
        Repo's main folder path
    results_dir : str
        Model results folder
    model_names : list
        Choosen model enum list
    
    Returns
    -------
    
    config_paths: list
        Choosen models' config paths
    model_file_paths: list
        Choosen models' model file paths
    model_result_paths: list
        Save Model Results to paths list
    selected_model_names: list
        Choosen models' names
        
    """
    
    config_paths = []
    model_file_paths = []
    model_result_paths = []
    conf_paths = sorted(glob.glob(os.path.join(current_dir, 'configs', '*.py')))
    model_paths = sorted(glob.glob(os.path.join(current_dir, 'models', '*.pth')))
    model_dict = {'0': 'ATSS',
                  '1': 'CASCADE R-CNN',
                  '2': 'DEFORMABLE DETR',
                  '3': 'DETR',
                  '4': 'DOUBLEHEAD R-CNN',
                  '5': 'DYNAMIC R-CNN',
                  '6': 'FASTER R-CNN',
                  '7': 'FCOS',
                  '8': 'RETINANET',
                  '9': 'VARIFOCALNET',
                  '10': 'YOLOv3',
                  }
    selected_model_names = []
    for i in model_names:
        config_paths.append(conf_paths[int(i)])
        model_file_paths.append(model_paths[int(i)])
        try:
            selected_model_names.append(model_dict[i])
        except KeyError:
            print("KeyError: model_enum parameter range must be between 0 and 10...")
            exit(-1)
        path_res = os.path.join(results_dir, os.path.basename(model_paths[int(i)]).split('.')[0])
        model_result_paths.append(path_res)
        if not os.path.exists(path_res):
            os.makedirs(path_res)
    return config_paths, model_file_paths, model_result_paths, selected_model_names


def get_image_annot_list(annot_path, img_path):
    """
    Getting annotations and image name paths.
    
    Parameters
    ---------- 
    annot_path : str
        Annotation path
    img_path : str
        Image path
    
    Returns
    -------
    
    img_list: list
        Image file names' list
    ann_list: list
        Image files' annotation list
        
    """
    
    ann_list = mmcv.list_from_file(annot_path)
    img_list = []
    for i, ann_line in enumerate(ann_list):
        if ann_line != '#':
            continue
        img_list.append(os.path.join(img_path, ann_list[i + 1]))
    return img_list, ann_list


def control_annot_path(annot_path, img_path):
    """
    Checking annotation path then if theres any annotation path calls get_image_annot_list 
    function otherwise it gets image file paths.
    
    Parameters
    ---------- 
    annot_path : str
        Annotation path
    img_path : str
        Image path
    
    Returns
    -------
    
    img_list: list
        Image file names' list
    ann_list: list
        Image files' annotation list
        
    """
    
    if annot_path:
        img_list, ann_list = get_image_annot_list(annot_path, img_path)
    else:
        img_list = sorted(glob.glob(os.path.join(img_path, '*')))
        ann_list = None
    return img_list, ann_list


def breast_segmentation(seg_model, image):
    """
    Extracts breast region from given Image with segmentation model.
    
    Parameters
    ---------- 
    seg_model : model object
        Developed segmentation model
    image : array
        Image
    
    Returns
    -------
    
    crop_img: array
        Cropped Image array
    crop_coordinate: list
        Cropped regions' coordinates [x_min, y_min, x_max, y_max]
        
    """
    
    image = rescale(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(image)
    seg_img = resize_and_rescale_sequence(clahe_image)
    seg_img = seg_img.reshape(1,512,512,1)
    seg_img = np.moveaxis(seg_img, 3, 1)
    seg_img = torch.from_numpy(seg_img).float()
    seg_img = seg_img.cuda()
    seg_output = seg_model(seg_img)
    output = output_to_mask(seg_output)
    abc = flood_fill(output[0][0])

    regions = mam_pre.get_image_regions(abc)
    cleared_region = mam_pre.clear_unnecessary_regions(regions, abc)
    clean_mask = mam_pre.resize_rescale_image(cleared_region, image.shape[0], image.shape[1])
    pos = np.where(clean_mask)
    xmin = pos[0].min()
    xmax = pos[0].max()
    ymin = pos[1].min()
    ymax = pos[1].max()
    crop_img = image[xmin:xmax, ymin:ymax]
    nxmin = ymin
    nxmax = ymax
    nymin = xmin
    nymax = xmax
    crop_coordinate = [nxmin, nymin, nxmax, nymax]
    crop_img = rescale(crop_img)
    return crop_img, crop_coordinate


def apply_segmentation(seg_img_path, img_path):
    """
    Extracts breast region from given Image with segmentation model.
    
    Parameters
    ---------- 
    seg_img_path : str
        Segmentation image path
    img_path : array
        Image
    
    Returns
    -------
    
    crop_coordinates: list
        Cropped regions' coordinates list
    img_shapes: list
        Cropped regions' shapes
        
    """
    
    img_list = sorted(glob.glob(os.path.join(img_path, '*')))
    path = os.path.join(os.path.sep, 'workspace','notebooks','new_mammo_repo','utils','ResUNet_breast.pth')
    seg_model = torch.load(path)
    print('Applying segmentation to images')
    crop_coordinates = []
    img_shapes = []
    for j in tqdm(range(len(img_list)), desc='Breast Segmentations'):
        image = cv2.imread(img_list[j])
        image, crop_coordinate = breast_segmentation(seg_model, image)
        img_file_name = os.path.basename(img_list[j])
        cv2.imwrite(os.path.join(seg_img_path, img_file_name), image)
        crop_coordinates.append(crop_coordinate)
        img_shapes.append(image.shape)
    return crop_coordinates, img_shapes
        
def get_gtbbox(ann_list):
    """
    Getting bounding box informations from annotation.

    Parametreler
    ---------- 
    ann_list : list
        Annotation List
    
    Returns
    -------
    bound_box : list
        Annotation ground truth bounding box list [xmin, ymin, xmax, ymax, label]
    ann_lists : list
        Annotation image file name list

    """
    
    end = True
    bound_box = []
    ann_lists = []
    i = 0
    while end:
        if ann_list[i] == "#":
            i+=1
            ann_lists.append(ann_list[i])
            i+=2
            count = int(ann_list[i])
            
        bb = []
        for j in range(count):
            i+=1
            bb.append(ann_list[i])

        bound_box.append(bb)

        if i == len(ann_list)-1:
            end = False

        i+=1
    return bound_box, ann_lists  


def get_annot_path(img_list, ann_list, img_shapes, crop_coordinates, seg_img_path):
    """
    Preparing annotation file for segmented images.

    Parametreler
    ---------- 
    img_list: list
        Image file names' list
    ann_list : list
        Annotation List
    img_shapes : list
        Images' shapes list
    crop_coordinates : list
        Cropped regions' coordinates list
    seg_img_path : str
        Segmentation image path
    
    Returns
    -------
    
    annot_path : str
        Segmented datas' annotation file path

    """
    
    bound_box, ann_lists = get_gtbbox(ann_list)
    n_img_list = []
    for imm in img_list:
        n_img_list.append(os.path.basename(imm))
    n_bound_box = []   
    for n_im in n_img_list:
        if n_im in ann_lists:
            n_bound_box.append(bound_box[ann_lists.index(n_im)])
    bound_box = n_bound_box 
    annot_path = os.path.join(seg_img_path, 'segmented_annotation.txt')
    for i in range(len(bound_box)):
        xmin, ymin, xmax, ymax = crop_coordinates[i]
        count = 0
        lines = []
        for j in range(len(bound_box[i])):
            parsed_val = bound_box[i][j].split()
            label = parsed_val[-1]
            coordinates = parsed_val[:-1]
            bxmin, bymin, bxmax, bymax = int(coordinates[0]),int(coordinates[1]),int(coordinates[2]), int(coordinates[3])       
            n_xmin = bxmin - xmin
            n_ymin = bymin - ymin
            n_xmax = n_xmin + (bxmax - bxmin)
            n_ymax = n_ymin + (bymax - bymin)
            bbox_label = [n_xmin, n_ymin, n_xmax, n_ymax, label]
            line = ' '.join(str(e) for e in bbox_label)
            lines.append(line)
            count += 1
            
        if len(lines)>0:

            with open(annot_path, 'a') as mask_output_file:
                mask_output_file.write("#"+"\n")
                mask_output_file.write(os.path.basename(img_list[i])+"\n")
                mask_output_file.write(str(img_shapes[i][1])+" "+str(img_shapes[i][0])+"\n")
                mask_output_file.write(str(count)+"\n")

                for z in range(len(lines)):
                    mask_output_file.write(lines[z]+"\n")
                mask_output_file.close()
                
    return annot_path


def create_result_dir(current_dir):
    """
    Creates results directory.

    Parametreler
    ---------- 
    current_dir : str
        Repo's main folder path
    
    Returns
    -------
    
    results_dir : str
        Model results folder

    """
    
    current_dir = os.path.join(current_dir, 'work_dirs')
    if os.path.isdir(current_dir):
        pass
    else:
        os.mkdir(current_dir)
    results_dir = os.path.join(current_dir, 'results', '')
    if os.path.isdir(results_dir):
        counter = 0
        while os.path.isdir(results_dir):
            results_dir = os.path.join(current_dir, 'results') + "{}"
            counter += 1
            results_dir = results_dir.format(counter)
    os.mkdir(results_dir)
    if not os.path.exists(os.path.join(results_dir, 'breast_segmentation', '')):
        os.makedirs(os.path.join(results_dir, 'breast_segmentation', ''))
    return results_dir


def write_labels_to_txt(label_names):
    """
    Preparing mammo_dataset.py appropriately.

    Parametreler
    ---------- 
    label_names : list
        Contains Classification [MALIGN, BENIGN] or Detection [MASS] labels.
    
    Returns
    -------
    

    """
    
    f = open(os.path.join(os.path.sep, 'workspace', 'notebooks', 'new_mammo_repo', 'classes.txt'), 'w')
    if len(label_names) == 1:
        f.write('MASS')
    elif len(label_names) == 2:
        f.write('MALIGN,BENIGN')
    f.close()


def merge_results(results):
    """
    Combines MALIGN and BENIGN detections if classify_mass parameter from terminal set as False.

    Parametreler
    ---------- 
    results : list
        model detections
    
    Returns
    -------
    new_results: list
        merge model detections

    """
    
    new_results = []
    for res in results:
        list_r = []
        for j in range(2):
            r = list(res[j])
            for a in r:
                list_r.append(a)
        if len(list_r)==0:
            rr = np.zeros((0,5)).astype('float32')
        else:
            rr = np.array(list_r)
        new_results.append([rr])
    return new_results


def get_model_predicts(config, checkpoint, img_list, class_size, device):
    """
    Getting models' detections

    Parametreler
    ---------- 
    config : str
        model config path
        
    checkpoint : str
        model file path
    
    img_list: list
        image file names' list
    
    class_size : int
        length of label names
        
    device : str
        set cuda device
    
    Returns
    -------
    results: list
        model detections

    """
    
    results = []
    model = init_detector(config, checkpoint, device)
    for j in img_list:
        result = inference_detector(model, j)
        results.append(result)
    if class_size==1:
        results = merge_results(results)
    return results


def normalize_bbox(bbox, shapex, shapey):
    """
    Process of normalizing the bbox points of the image for NMS.

    Parametreler
    ----------    
    bbox: list
        images' bounding box

    Returns
    -------
    bbox: list
        normalized bounding box

    """
    bbox = bbox.copy()
    bbox[0] /= shapex
    bbox[1] /= shapey
    bbox[2] /= shapex
    bbox[3] /= shapey
    return bbox


def de_normalize_bbox(bbox, shapex, shapey):
    """
    Process of denormalizing the normalized bbox points of the image for NMS.
    

    Parametreler
    ---------- 
    bbox: list
        normalized bbox

    Returns
    -------
    bbox: list
        bbox

    """
    bbox = bbox.copy()
    bbox[0] *= shapex
    bbox[1] *= shapey
    bbox[2] *= shapex
    bbox[3] *= shapey
    return bbox


def get_labels_scores_boxes_list(results, shapex, shapey, class_size):
    """
    Extraction of label, score, box lists in accordance with 
    each image of the result list obtained in the model output.


    Parametreler
    ----------    
    results: list
        List of model results
    shapex: int
        image weight
    shapey: int
        image height
    class_size : int
        length of label names

    Returns
    -------
    all_labels_list: list
        list of labels
    all_scores_list: list
        list of scores
    all_boxes_list: list
        list of boxes

    """
    
    all_labels_list = []
    all_scores_list = []
    all_boxes_list = []
    for j in range(len(results)):
        labels_list = []
        scores_list = []
        boxes_list = []
        for i in range(class_size):
            if(results[j][i].shape[0]>0):
                [labels_list.append(i) for k in range(results[j][i].shape[0])]
                scores_list.append(results[j][i][:,-1])
                for k in results[j][i][:,:-1]:
                    boxes_list.append(normalize_bbox(k, shapex, shapey))
        if len(scores_list) > 0: 
            all_labels_list.append(np.array(labels_list))
            all_scores_list.append(np.concatenate(scores_list))
            all_boxes_list.append(boxes_list)
    return all_labels_list,all_scores_list,all_boxes_list


def apply_nms(result, image_path, class_size, iou_thr=0.1, scr_thr=0):
    """
    Apply NMS for detection results

    Parametreler
    ---------- 
    result: list
        model detection result
    image_path: str
        image path
    class_size : int
        length of label names
    iou_thr: int
        iou threshold
    scr_thr: int
        confidence threshold

    Returns
    -------
    nms_results: list
        NMS applied results

    """
    
    img = Image.open(image_path)
    img = np.array(img)
    labels_list,scores_list,boxes_list = get_labels_scores_boxes_list([result], img.shape[1], img.shape[0], class_size)
    try:
        b, s, l = nms(boxes_list, scores_list, labels_list, iou_thr=iou_thr)
        # b, s, l = soft_nms(boxes_list, scores_list, labels_list, iou_thr=iou_thr, thresh=scr_thr)
    except ValueError:
        b = []
        s = []
        l = []
    
    n_boxes = []
    for y in range(len(b)):
        n_boxes.append(de_normalize_bbox(b[y], img.shape[1], img.shape[0]))
    
    label_dict = {}
    for clss in range(class_size):
        label_dict[str(clss)] = []
    
    for n in range(len(n_boxes)):
        if s[n] > scr_thr:
            list_b = list(n_boxes[n])
            list_b.append(s[n])
            arr_b = np.array(list_b) 
            label_dict[str(l[n])].append(arr_b)
    
    nms_results = []
    for key, value in label_dict.items():           
        r = np.array(value)
        if len(r)==0:
            r = np.zeros((0, 5)).astype('float32')
        nms_results.append(r)
    return nms_results


def get_nms_results(results, img_list, class_size, iou_thr, scr_thr):
    """
    Getting NMS applied results

    Parametreler
    ---------- 
    results: list
        model detection result
    img_list: list
        images' path list
    class_size : int
        length of label names
    iou_thr: int
        iou threshold
    scr_thr: int
        confidence threshold

    Returns
    -------
    nms_results: list
        NMS applied results

    """
    
    print('--> Applying Non Maximum Suppression to model predicts...')
    nms_results = []
    for i in range(len(img_list)):
        nms_res = apply_nms(results[i], img_list[i], class_size, iou_thr, scr_thr)
        nms_results.append(nms_res)
    return nms_results


def get_ensemble_results_format(boxes, scores, labels, class_size):
    """
    Preparing results for ensemble format

    Parametreler
    ---------- 
    boxes: list
        bounding boxes list
    scores: list
        scores list
    labels: list
        labels list
    class_size : int
        length of label names

    Returns
    -------
    ensemble_results: list
        results at ensemble format

    """
    
    ensemble_results=[]
    [ensemble_results.append([]) for i in range(class_size)]
    
    for i in range(len(labels)):
        ensemble_results[int(labels[i])].append(list(np.concatenate((boxes[i],[scores[i]]))))
    for i in range(class_size):
        if(len(ensemble_results[i])>0):
            ensemble_results[i] = np.array(ensemble_results[i])
        else:
            ensemble_results[i] = np.empty([0,5])
    return ensemble_results


def calculate_ious(ground_truth, pred):
    """
    Calculate iou metric between ground truth and detection.

    Parametreler
    ---------- 
    ground_truth : list
        ground_truth bounding box 
    pred : list
        detection bounding box 
    
    Returns
    -------
    iou : float
        iou value between ground truth and detection

    """
    x1 = np.maximum(ground_truth[0], pred[0])
    y1 = np.maximum(ground_truth[1], pred[1])
    x2 = np.minimum(ground_truth[2], pred[2])
    y2 = np.minimum(ground_truth[3], pred[3])
    
    # Intersection height and width.
    height = np.maximum(y2 - y1 + 1, np.array(0.))
    width = np.maximum(x2 - x1 + 1, np.array(0.))
    
    area_of_intersection = height * width
    
    # Ground Truth dimensions.
    gt_height = ground_truth[3] - ground_truth[1] + 1
    gt_width = ground_truth[2] - ground_truth[0] + 1
    
    # Prediction dimensions.
    pd_height = pred[3] - pred[1] + 1
    pd_width = pred[2] - pred[0] + 1
    
    area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection
    
    iou = area_of_intersection / area_of_union
    
    return iou


def filter_results(results, iou_class_thr=0.3):
    """
    Filtering results by selecting the one with the highest confidence 
    score when predicting both benign and malignant to an audience

    Parametreler
    ---------- 
    results: list
        model detections
    iou_class_thr: float    
        iou value between different classes
    
    Returns
    -------
    filter_results : list
        filtered results

    """
    
    filter_results = []
    for ens in results:
        boxes = []
        scores = []
        labels = []
        for cl in range(len(ens)):
            for pr in ens[cl]:
                boxes.append(pr[:-1])
                scores.append(pr[-1])
                labels.append(cl)

        extract_index = []
        len_boxes = len(boxes)
        for f in range(len_boxes):
            for f1 in range(f+1, len_boxes, 1):
                iou = calculate_ious(boxes[f], boxes[f1])
                if iou > iou_class_thr:
                    if scores[f] > scores[f1]:
                        extract_index.append(f1)
                    else:
                        extract_index.append(f)
        extract_index = sorted(list(set(extract_index)))
        extract_index.reverse()

        for inx in extract_index:
            boxes.pop(inx)
            scores.pop(inx)
            labels.pop(inx)

        filter_results.append(get_ensemble_results_format(boxes, scores, labels, 2))
    return filter_results


def apply_scr_threshold(results, class_size, conf_thresh=0.05):
    """
    Applying score threshold to model detections

    Parametreler
    ---------- 
    results: list
        model results
    class_size : int
        length of label names
    conf_thresh: int
        confidence threshold

    Returns
    -------
    all_boxes_list: list
        list of boxes
    all_scores_list: list
        list of scores

    """
    all_scores_list = []
    all_boxes_list = []
    for j in range(len(results)):
        scores_list = []
        boxes_list = []
        for i in range(class_size):
            bbboxes = []
            conf_scr = []
            boxes_list.append([])
            scores_list.append([])
            for k in range(len(results[j][i][:,:-1])):
                conf_value = float(results[j][i][:,-1][k])
                if conf_value > conf_thresh:
                    conf_scr.append(results[j][i][:,-1][k])
                    bbboxes.append(results[j][i][:,:-1][k])
            boxes_list[i] = bbboxes
            scores_list[i] = conf_scr
        all_scores_list.append(scores_list)  # np.concatenate(scores_list)
        all_boxes_list.append(boxes_list)
    return all_boxes_list, all_scores_list


def get_result_array(results_boxes, results_scores):
    """
    Creating results array from bbox and score list

    Parameters
    ----------
    results_boxes : list
        detection bound box listesi 
    results_scores : list
        detection score listesi
    
    Returns
    -------
    results : list
        detection list

    """
    class_length = len(results_boxes[0])
    results = []
    for i in range(len(results_boxes)):
        result = []
        for k in range(class_length):
            boxss = np.empty(shape=[len(results_boxes[i][k]),5], dtype="float32")
            for j in range(len(results_boxes[i][k])):
                detection = results_boxes[i][k][j]
                detection = np.append(detection,results_scores[i][k][j])
                
                boxss[j] = detection
            result.append(boxss)
        results.append(result)   
    return results


def get_class_gts(ground_truth, class_size):
    """
    Getting bounding box coordinates

    Parameters
    ----------
    ground_truth : list
        ground truth list with scores and coordinates
    class_size : int
        length of label names
    
    Returns
    -------
    ground_truths : list
        ground truth bound box coordinate list

    """
    
    ground_truths = []
    for i in range(class_size):
        ground_truths.append([])
    for i in range(len(ground_truth)):
        xmin,ymin,xmax,ymax,label = ground_truth[i].split()
        ground_truths[int(label)].append([xmin,ymin,xmax,ymax])
    return ground_truths


def get_tp_fp_fn_classwises(result, gt, ap_threshold, class_size=1):
    """
    Calculation of TP, FP, FN for each class.

    Parameters
    ----------
    result: list
        model detections
    gt : list
        ground truth list with scores and coordinates
    ap_threshold : float
        TP iou threshold
    class_size : int
        length of label names
    
    Returns
    -------
    tp : numpy array
        True positive
    fp : numpy array
        False positive
    fn : numpy array
        False negative

    """

    tp = np.zeros((1,class_size))
    fp = np.zeros((1,class_size))
    fn = np.zeros((1,class_size))
    
    gt_cnt = np.zeros((1,class_size))
    detect_cnt = np.zeros((1,class_size))
    for i in range(class_size):
        gt_cnt[0][i] = len(gt[i])
        detect_cnt[0][i]= len(result[i])
    confidences = []
    detections_indexes = []
    for k in range(class_size):
        detected_indexes = [0]*len(result[k])
        for j in range(len(gt[k])):
            bb_coordinate = list(map(int, gt[k][j]))
            ious = []
            for i in range(len(result[k])):
                if detected_indexes[i] == 1:
                    iou = 0
                else:
                    det = result[k][i]
                    pred_coordinate = list(map(int, det[:-1]))
                    iou = calculate_ious(bb_coordinate,pred_coordinate)
                ious.append(iou)
            if ious:
                max_iou = max(ious)
                if max_iou >= ap_threshold and len(result[k])!=0:
                    tp[0][k]+=1
                    gt_cnt[0][k]-=1
                    detect_cnt[0][k]-=1
                    max_iou_index = ious.index(max_iou)
                    detected_indexes[max_iou_index]+=1
        detections_indexes.append(detected_indexes)
    gt_length = np.zeros((1,class_size))
    det_length = np.zeros((1,class_size))
    for i in range(class_size):
        gt_length[0][i] = len(gt[i])
        det_length[0][i] = len(result[i])
        fn[0][i] = gt_cnt[0][i]
        fp[0][i] = det_length[0][i] - tp[0][i]
    
    return tp,fp,fn   
    
    
def get_recall_fppi(config, ann_list, img_list, results, ap_threshold, step_threshold, label_names):
    """
    Getting Recall and FPPI metrics

    Parameters
    ----------
    config: str
        config path
    ann_list : list
        Annotation List
    img_list: list
        Image file names' list
    results: list
        model detections
    ap_threshold : float
        TP iou threshold
    step_threshold : int
        score threshold step size
    label_names : list
        Contains Classification [MALIGN, BENIGN] or Detection [MASS] labels.
    
    Returns
    -------
    recalls : list
        Recalls for each class for each confidence
    fppis : list
        FPPIs for each class for each confidence

    """
    fppis = []
    recalls = []
    class_size = len(label_names)
    bound_box, _ = get_gtbbox(ann_list)
    print('--> Calculating RECALL, FPPI metrics for each confidence threshold for FROC CURVE...')
    for i in range(step_threshold):
        filtered_conf_thr = i/step_threshold
        
        conf_filter_boxes, conf_filter_scores = apply_scr_threshold(results, class_size, filtered_conf_thr)
        new_results = get_result_array(conf_filter_boxes, conf_filter_scores)

        tps = np.zeros((1,class_size))
        fps = np.zeros((1,class_size))
        fns = np.zeros((1,class_size))
        for j in range(len(new_results)):
            result = new_results[j]
            ground_truth = get_class_gts(bound_box[j], class_size)
            tp, fp, fn = get_tp_fp_fn_classwises(result, ground_truth, ap_threshold, class_size=class_size)
            tps += tp
            fps += fp
            fns += fn

        gt_size = tps+fns
        detect_size = tps+fps
        recall = tps/gt_size
        
        for j in range(class_size):
            if gt_size[0][j] == 0:
                recall[0][j] = 0

        fppi = fps/len(img_list)
        fppis.append(fppi)
        recalls.append(recall)
        
    return recalls, fppis


def get_result_dict(results, ann_list, ap_threshold, class_size):
    """
    Getting num_gts, num_dets, ap, mAP metrics

    Parameters
    ----------
    results: list
        model detections
    ann_list : list
        Annotation List
    ap_threshold : float
        TP iou threshold
    class_size : int
        length of label names
    
    Returns
    -------
    num_gts : list
        ground truth length for each class
    num_dets : list
        detection length for each class
    ap : list
        average precision for each class
    mAP : list
        mean average precision
    """
    
    annotations, _ = get_gtbbox(ann_list)

    lis=[]
    for ann in annotations:
        annot_dict = {
            'bboxes': [],
            'labels': []}
        for a in ann:
            parsed = a.split()
            annot_dict['labels'].append(int(parsed[-1]))
            annot_dict['bboxes'].append([int(parsed[0]), int(parsed[1]), int(parsed[2]), int(parsed[3])])
        annot_dict['bboxes'] = np.array(annot_dict['bboxes'])
        annot_dict['labels'] = np.array(annot_dict['labels'])
        lis.append(annot_dict)

    annotations = lis

    mAP, result_dict = eval_map(results,
                 annotations,
                 scale_ranges=None,
                 iou_thr=ap_threshold,
                 ioa_thr=None,
                 dataset=None,
                 logger='silent',
                 tpfp_fn=None,
                 nproc=4,
                 use_legacy_coordinate=False,
                 use_group_of=False)
    num_gts = []
    num_dets = []
    ap = []
    for i in range(class_size):
        num_gts += [result_dict[i]['num_gts']]
        num_dets += [result_dict[i]['num_dets']]
        ap += [result_dict[i]['ap']]
    return num_gts, num_dets, ap, mAP


def model_evals(config_paths, results, ap_threshold, img_path, annot_path, class_size):
    """
    Evaluation of model results

    Parametreler
    ---------- 
    config_paths : list
        Choosen models' config paths
    results: list
        model detections
    ap_threshold : float
        TP iou threshold
    img_path : str
        Image path
    annot_path : str
        Annotation path
    class_size : int
        length of label names
    Returns
    -------
    mean_ap : list
        mean average precision

    """
    cfg = Config.fromfile(config_paths)
    cfg.data.test.test_mode = True
    cfg.data.test['ann_file'] = annot_path
    cfg.data.test['img_prefix'] = img_path
    if class_size == 1:
        cfg.classes = ('MASS',)
        # cfg.custom_imports = dict(imports=['mammo_dataset_mass'], allow_failed_imports=False)
        # cfg.dataset_type = 'MASS_MammoDataset'
        # cfg.data.test.type = 'MASS_MammoDataset'
    else:
        cfg.classes = ('MALIGN', 'BENIGN')
        # cfg.custom_imports = dict(imports=['mammo_dataset'], allow_failed_imports=False)
    dataset = build_dataset(cfg.data.test)
    outputs = results

    kwargs = {}

    eval_kwargs = cfg.get('evaluation', {}).copy()
    # hard-code way to remove EvalHook args
    for key in [
            'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
            'rule'
    ]:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metric="mAP", **kwargs))

    
    y = dataset.evaluate(outputs, iou_thr=ap_threshold, **eval_kwargs)
    mean_ap = y['mAP']
    return mean_ap


# def get_result_metrics(results, config_path, img_list, ann_list, label_names, selected_model, ap_threshold, img_path, annot_path, confidence_threshold, results_dict):
#     """
#     Applying ensemble to all choosen models

#     Parametreler
#     ---------- 
#     results: list
#         model detections
#     config_paths : list
#         Choosen models' config paths
#     img_list: list
#         Image file names' list
#     ann_list: list
#         Image file names' list
#     label_names : list
#         Contains Classification [MALIGN, BENIGN] or Detection [MASS] labels.
#     selected_model: list
#         Choosen model's name
#     ap_threshold : float
#         TP iou threshold
#     img_path : str
#         Image path
#     annot_path : str
#         Annotation path
#     confidence_threshold : float
#         score threshold
#     results_dict: dict
#         dictionary for result metrics
   
#     Returns
#     -------
#     results_dict: dict
#         dictionary for result metrics

#     """
#     class_size = len(label_names)
#     recalls, fppis = get_recall_fppi(config_path, ann_list, img_list, results, ap_threshold, 100, label_names)
#     recall_list.append(recalls)
#     fppis_list.append(fppis)
#     num_gts, num_dets, ap, mAP = get_result_dict(results, ann_list, ap_threshold, class_size)
#     model_evals(config_path, results, ap_threshold, img_path, annot_path, class_size)
#     results_dict[selected_model] = [[num_gts], [num_dets], recalls[0], fppis[0], ap, confidence_threshold, mAP]
#     return results_dict 


def applying_ensemble(img_list, model_results, class_size, iou_thr, scr_thr):
    """
    Applying ensemble to all choosen models

    Parametreler
    ---------- 
    img_list: list
        Image file names' list
    model_results: list
        all model detections 
    class_size : int
        length of label names
    iou_thr : float
        iou threshold
    scr_thr : float
        score threshold
   
    Returns
    -------
    ensemble_bbox : list
        ensemble detections result

    """
    
    ensemble_bbox = []
    for i in range(len(img_list)):
        img = Image.open(img_list[i])
        img = np.array(img)
        predictions = []
        for j in range(len(model_results)):
            predictions.append(model_results[j][0][i])
        labels_list,scores_list,boxes_list = get_labels_scores_boxes_list(predictions, img.shape[1], img.shape[0], class_size=class_size)
        boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, iou_thr=iou_thr, skip_box_thr=scr_thr)
        n_boxes = []
        for y in range(len(boxes)):
            n_boxes.append(de_normalize_bbox(boxes[y], img.shape[1], img.shape[0]))
        ensemble_bbox.append(get_ensemble_results_format(n_boxes, scores, labels, class_size))
    ensemble_bbox = get_nms_results(ensemble_bbox, img_list, class_size, iou_thr, scr_thr=scr_thr)
    return ensemble_bbox
    
    
def save_results(img_list, ann_list, results, label_names, img_save_path):
    """
    Saving detection results on image (if annotations are exist, they will be saved on image)

    Parametreler
    ---------- 
    img_list: list
        Image file names' list
    ann_list: list
        Image files' annotation list 
    results: list
        model detections
    label_names : list
        Contains Classification [MALIGN, BENIGN] or Detection [MASS] labels.
    img_save_path : str
        save path
   
    Returns
    -------

    """
    
    class_size = len(label_names)
    print('--> Detections and their annotations (if you give any annotation path for it) showed on the image and images are saving to', img_save_path, 'folder...')
    if ann_list:
        bound_box, _ = get_gtbbox(ann_list)
    for index in tqdm(range(len(img_list)), desc='Save Results'):
        img = cv2.imread(img_list[index])
        if ann_list:
            for i in range(len(bound_box[index])):
                parsed_val = bound_box[index][i].split()
                label = parsed_val[-1]
                text = label_names[int(label)]
                coordinates = parsed_val[:-1]
                xmin,ymin,xmax,ymax = int(coordinates[0]),int(coordinates[1]),int(coordinates[2]), int(coordinates[3])
                img = cv2.rectangle(img, (xmin,ymin),(xmax,ymax),(0,255,0), thickness=4)
                img = cv2.putText(img, text, (xmin,ymax+40), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1.5, color=(128,255,128), thickness = 3)
        
        for i in range(class_size):
            if results[index][i] == []:
                continue
            for res in results[index][i]:
                coordinates = res
                xmin,ymin,xmax,ymax = int(coordinates[0]),int(coordinates[1]),int(coordinates[2]), int(coordinates[3])
                if i==0:
                    img = cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (255, 0, 0 ), thickness=8)
                    img = cv2.putText(img, label_names[i] + "-{:.2f}".format(coordinates[4]), (xmin,ymin-20), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1.5, color=(255, 0, 0), thickness = 3)
                else:
                    img = cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (0,0,139), thickness=8)
                    img = cv2.putText(img, label_names[i] + "-{:.2f}".format(coordinates[4]), (xmin,ymin-20), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color=(0,0,139), thickness = 3)
        
        img_result_path = os.path.join(img_save_path, os.path.basename(img_list[index]))
        plt.imsave(img_result_path, img)
        
        
def save_metrics_to_csv(results_dir, results_dict, label_names):
    """
    Saving result metrics as csv.

    Parametreler
    ---------- 
    results_dir : str
        Model results folder
    results_dict: dict
        dictionary for result metrics
    label_names : list
        Contains Classification [MALIGN, BENIGN] or Detection [MASS] labels.
   
    Returns
    -------

    """
    
    class_size = len(label_names)
    result_keys = list(results_dict.keys())

    row_values = []
    results_dict_keys = ['Ground Truth Object Count', 'Detection Object Count', 'TPR', 'FPPI']
    for i in range(len(result_keys)): #4 model vari
        for z in range(class_size):
            row = {}
            row['Models'] = result_keys[i]
            row['Confidence Threshold'] = results_dict[result_keys[i]][5]
            row['Classes'] = label_names[z]
            
            for a in range(len(results_dict_keys)): 
                row[results_dict_keys[a]] = round(results_dict[result_keys[i]][a][0][z],3)
            row['AP'] = round(results_dict[result_keys[i]][4][z],3)
            row['mAP'] = round(results_dict[result_keys[i]][-1],3)
            row_values.append(list(row.values()))
    csv_path = os.path.join(results_dir, 'results.csv')
    row_df = pd.DataFrame(row_values, columns = list(row.keys()))
    row_df.set_index(['Models', 'Confidence Threshold', 'Classes', 'Ground Truth Object Count', 'Detection Object Count', 'TPR', 'FPPI', 'AP', 'mAP'])
    row_df.to_csv(csv_path, index=False)
    print('Results File include metrics of: Models | Confidence Threshold | Classes | Ground Truth Count | Detection Count | TPR | FPPI | AP |  mAP')
    print('--> Metrics saved to ', csv_path)
    

def get_faucs(recalls, fppis, class_size):
    """
    Calculate FAUC score (Area Under FROC)

    Parameters
    ----------
    recalls : list
        Recall list
    fppis : list
        FPPI list
    class_size : int
        sınıf sayısı
    
    Returns
    -------
    faucs : list
        FAUCs value of all models

    """
    
    faucs = [[] for i in range(class_size)]
    fpss = [[] for i in range(class_size)]
    for l in range(len(recalls)):
        fp_te = [0] * class_size
        for r in range(len(recalls[l])):
            for c in range(class_size):
                if fppis[l][r][0][c] > fp_te[c]:
                    fp_te[c] = fppis[l][r][0][c]
        for c in range(class_size): 
            fpss[c].append(fp_te[c])

    for l in range(len(recalls)):
        tprs = [[] for i in range(class_size)]
        fppis_list = [[] for i in range(class_size)]
        for r in range(len(recalls[l])):
            for c in range(class_size):
                tprs[c].append(recalls[l][r][0][c])
                fppis_list[c].append(fppis[l][r][0][c])

        for t in range(len(tprs)):
            tprs[t].insert(0, max(tprs[t]))
            fppis_list[t].insert(0, max(fpss[t]))
            fps = np.array(fppis_list[t])/max(fpss[t])
            fauc = auc(fps,tprs[t])
            faucs[t].append(round(fauc,3))
    return faucs


def plot_froc(recalls, fppis, faucs, label_names, model_names, figsize=(10,8), dpi=100, save_fig_path=None):
    """
    Plotting FROC

    Parameters
    ----------
    recalls : list
        Recalls lists
    fppis : list
        FPPIs lists
    label_names : list
        Contains Classification [MALIGN, BENIGN] or Detection [MASS] labels.
    model_names : list
        Choosen model enum list
    figsize : tuple
        figure size
    dpi : int
        figure resolution
    save_fig_path : str
        figure save path
        
    Returns
    -------
    class_lines : list
        line lists
    """
    class_size = len(label_names)
    for k in range(class_size):
        plt.figure(figsize=figsize, dpi=dpi)
        for rf in range(len(recalls)):
            xs = [x[0][k] for x in fppis[rf]]
            ys = [x[0][k] for x in recalls[rf]]
            line, = plt.plot(xs, ys, label=model_names[rf]+" (FAUC Score: "+str(faucs[k][rf])+")")
            plt.title("FROC Curve for " + label_names[k])
            plt.xlabel("False Positive per Image")
            plt.ylabel("True Positive Rate")
            plt.xlim(left=-0.001)
            plt.ylim([0,1.001])
            plt.legend(loc="lower right", prop={'size': 10})
            plt.grid(True)

        if save_fig_path:
            
            plt.rcParams['savefig.facecolor']='white'
            plt.savefig(os.path.join(save_fig_path, label_names[k] + "_froc.png"))
            print('--> ' + label_names[k] + "'s FROC figure saved...")
            plt.clf()

    






