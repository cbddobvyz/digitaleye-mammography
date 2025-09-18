<div align="center" markdown="1">
<img src="docs/sayısal-göz-banner2.jpeg" />

**<center><h1> DIGITAL EYE for MAMMOGRAPHY </h1></center>**

**The Digital Eye for Mammography: Deep Transfer Learning and Model Ensemble based Open-Source Toolkit for Mass Detection and Classification.** [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=The%20Digital%20Eye%20for%20Mammography%20:%20Deep%20Transfer%20Learning%20and%20Model%20Ensemble%20based%20Open-Source%20Toolkit%20for%20Mass%20Detection%20and%20Classification%20repository.&url=https://github.com/cbddobvyz/digitaleye-mammography&via=dijital&hashtags=DigitalEye,AI,BreastCancer,EarlyDiagnosis,OpenSource)
</div>

<div align="center">
<p align="center">
  <a href="https://cbddo.gov.tr/en/projects/digital-eye-project/">Website</a> •
  <a href="#benchmarks">Benchmarks</a> •
  <a href="#installation">Installation</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#acknowledgements">Acknowledgements</a> •
  <a href="#citation">Citation</a> •
  <a href="#license">License</a> •
  <a href="#disclaimer">Disclaimer</a> •
</p>
 <p align="center">
  <img src="https://img.shields.io/badge/python-3.8-blue" />
  <img src="https://img.shields.io/badge/pytorch-1.12.1-blue" />
  <img src="https://img.shields.io/badge/mmdetection-2.28.2-blue" />
  <img src="https://img.shields.io/badge/ultralytics-8.3.6-blue" />
  <a href="https://github.com/ddobvyz/digitaleye-mammography/releases/tag/shared-models.v1" ><img src="https://img.shields.io/badge/pre--trained%20models-31-brightgreen" /></a>
  <a href="https://github.com/ddobvyz/digitaleye-mammography/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-GNU/GPLv3-blue" /></a>
 </p>   
</div>   

## **Introduction**

Breast cancer is both the most prevalent type of cancer across the world and a malignant disease with the highest rate of cancer-related mortality among women. Breast cancer comes first among the most prevalent types of cancer in TÜRKİYE.

As the Digital Transformation Office of the Presidency of the Republic of Türkiye, we are carrying out the Digital Eye Project (Breast Cancer Detection with Artificial Intelligence) in order to assist radiologists in screenings with mammography applied for the early diagnosis of breast cancer and to reduce their workload. 

![image](docs/model_result.jpg)

Key features:

* **Powered by Carefully Prepared Data:** This project is built on meticulously prepared and labeled KETEM dataset.

* **Transfer Learning Made Easy:** You can use this repository as a source for transfer learning, making it easier to leverage pre-trained models and adapt them to new tasks.

* **Boost Performance with Ensemble Models:** With various strategies, this project allows you to combine outputs from different deep learning architectures, enhancing the overall performance.

* **Compatible with MMDetection and Ultralytics:** This project plays well with [MMdetection](https://github.com/open-mmlab/mmdetection) and [Ultralytics](https://docs.ultralytics.com/), making it effortless to use and open to developers for adding new features.

* **Visualize and Compare Model Results:** Get a visual representation of your model's performance and generate detailed comparison reports. Calculate scientific metrics like True Positive Rate (TPR), Average Precision (AP) for each class, and mean Average Precision (mAP).

* **Open-Source and Accessible:** This toolkit is available as an open-source project, fostering collaboration and enabling developers to contribute and benefit from its features.

## **Benchmarks**

### **Ultralytics YOLO Benchmarks**

Benchmark of YOLO object detection models (<b>YOLOv8</b>, <b>YOLOv9</b>, <b>YOLOv10</b>, <b>YOLOv11</b>) which trained on the private mammography dataset KETEM are available in the below. Models can also be downloaded from [the releases.](https://github.com/ddobvyz/digitaleye-mammography/releases/tag/shared-models.v2)
    
#### **YOLOv11 Benchmarks**

|   Model | Class | TPR | FPPI | AP | mAP | Checkpoints |
| :------: | :-----------: | :-------: | :-----------: | :---------------: | :---------: | :---------: |
| <b> YOLO11n  | BI-RADS 4-5 <br> BI-RADS 1-2 | 0.765 <br> 0.764 | 0.076 <br> 0.382 | 0.867 <br> 0.815 | 0.841  | [checkpoint](https://github.com/ddobvyz/digitaleye-mammography/releases/download/shared-models.v2/yolo11_n.pt) |
| <b> YOLO11s | BI-RADS 4-5 <br> BI-RADS 1-2 | 0.8 <br> 0.757 | 0.086 <br> 0.355 | 0.864 <br> 0.82 | 0.842 | [checkpoint](https://github.com/ddobvyz/digitaleye-mammography/releases/download/shared-models.v2/yolo11_s.pt) |
| <b> YOLO11m | BI-RADS 4-5 <br> BI-RADS 1-2 | 0.799 <br> 0.776 | 0.069 <br> 0.357 | 0.881 <br> 0.825 | 0.853  | [checkpoint](https://github.com/ddobvyz/digitaleye-mammography/releases/download/shared-models.v2/yolo11_m.pt) |
| <b> YOLO11l | BI-RADS 4-5 <br> BI-RADS 1-2 | 0.811 <br> 0.772 | 0.091 <br> 0.325 | 0.88 <br> 0.834 | 0.857 | [checkpoint](https://github.com/ddobvyz/digitaleye-mammography/releases/download/shared-models.v2/yolo11_l.pt) |
| <b> YOLO11x | BI-RADS 4-5 <br> BI-RADS 1-2 | 0.794 <br> 0.766 | 0.07 <br> 0.358 | 0.88 <br> 0.831 | 0.856 | [checkpoint](https://github.com/ddobvyz/digitaleye-mammography/releases/download/shared-models.v2/yolo11_x.pt) |
    
#### **YOLOv10 Benchmarks**
  
|   Model | Class | TPR | FPPI | AP | mAP | Checkpoints |
| :------: | :-----------: | :-------: | :-----------: | :---------------: | :---------: | :---------: |
| <b> YOLOv10n  | BI-RADS 4-5 <br> BI-RADS 1-2 | 0.794 <br> 0.73 | 0.095 <br> 0.292 | 0.856 <br> 0.810 | 0.833  | [checkpoint](https://github.com/ddobvyz/digitaleye-mammography/releases/download/shared-models.v2/yolo10_n.pt) |
| <b> YOLOv10s | BI-RADS 4-5 <br> BI-RADS 1-2 | 0.789 <br> 0.741 | 0.086 <br> 0.316 | 0.857 <br> 0.814 | 0.836 | [checkpoint](https://github.com/ddobvyz/digitaleye-mammography/releases/download/shared-models.v2/yolo10_s.pt) |
| <b> YOLOv10m | BI-RADS 4-5 <br> BI-RADS 1-2 | 0.802 <br> 0.748 | 0.08 <br> 0.306 | 0.864 <br> 0.826 | 0.845  | [checkpoint](https://github.com/ddobvyz/digitaleye-mammography/releases/download/shared-models.v2/yolo10_m.pt) |
| <b> YOLOv10l | BI-RADS 4-5 <br> BI-RADS 1-2 | 0.818 <br> 0.764 | 0.066 <br> 0.297 | 0.874 <br> 0.826 | 0.85 | [checkpoint](https://github.com/ddobvyz/digitaleye-mammography/releases/download/shared-models.v2/yolo10_l.pt) |
| <b> YOLOv10x | BI-RADS 4-5 <br> BI-RADS 1-2 | 0.806 <br> 0.761 | 0.07 <br> 0.295 | 0.87 <br> 0.82 | 0.845 | [checkpoint](https://github.com/ddobvyz/digitaleye-mammography/releases/download/shared-models.v2/yolo10_x.pt) |

#### **YOLOv9 Benchmarks**
  
|   Model | Class | TPR | FPPI | AP | mAP | Checkpoints |
| :------: | :-----------: | :-------: | :-----------: | :---------------: | :---------: | :---------: |
| <b> YOLOv9t  | BI-RADS 4-5 <br> BI-RADS 1-2 | 0.789 <br> 0.765 | 0.075 <br> 0.37 | 0.873 <br> 0.819 | 0.846  | [checkpoint](https://github.com/ddobvyz/digitaleye-mammography/releases/download/shared-models.v2/yolo9_t.pt) |
| <b> YOLOv9s | BI-RADS 4-5 <br> BI-RADS 1-2 | 0.798 <br> 0.771 | 0.079 <br> 0.356 | 0.873 <br> 0.822 | 0.848 | [checkpoint](https://github.com/ddobvyz/digitaleye-mammography/releases/download/shared-models.v2/yolo9_s.pt) |
| <b> YOLOv9m | BI-RADS 4-5 <br> BI-RADS 1-2 | 0.784 <br> 0.777 | 0.077 <br> 0.346 | 0.87 <br> 0.836 | 0.853  | [checkpoint](https://github.com/ddobvyz/digitaleye-mammography/releases/download/shared-models.v2/yolo9_m.pt) |
| <b> YOLOv9c | BI-RADS 4-5 <br> BI-RADS 1-2 | 0.785 <br> 0.786 | 0.082 <br> 0.376 | 0.873 <br> 0.835 | 0.854 | [checkpoint](https://github.com/ddobvyz/digitaleye-mammography/releases/download/shared-models.v2/yolo9_c.pt) |
| <b> YOLOv9e | BI-RADS 4-5 <br> BI-RADS 1-2 | 0.798 <br> 0.777 | 0.084 <br> 0.345 | 0.879 <br> 0.839 | 0.859 | [checkpoint](https://github.com/ddobvyz/digitaleye-mammography/releases/download/shared-models.v2/yolo9_e.pt) |
    
#### **YOLOv8 Benchmarks**
  
|   Model | Class | TPR | FPPI | AP | mAP | Checkpoints |
| :------: | :-----------: | :-------: | :-----------: | :---------------: | :---------: | :---------: |
| <b> YOLOv8n  | BI-RADS 4-5 <br> BI-RADS 1-2 | 0.788 <br> 0.769 | 0.077 <br> 0.399 | 0.873 <br> 0.803 | 0.838  | [checkpoint](https://github.com/ddobvyz/digitaleye-mammography/releases/download/shared-models.v2/yolo8_n.pt) |
| <b> YOLOv8s | BI-RADS 4-5 <br> BI-RADS 1-2 | 0.805 <br> 0.774 | 0.089 <br> 0.373 | 0.871 <br> 0.825 | 0.848 | [checkpoint](https://github.com/ddobvyz/digitaleye-mammography/releases/download/shared-models.v2/yolo8_s.pt) |
| <b> YOLOv8m | BI-RADS 4-5 <br> BI-RADS 1-2 | 0.789 <br> 0.763 | 0.081 <br> 0.359 | 0.873 <br> 0.829 | 0.851  | [checkpoint](https://github.com/ddobvyz/digitaleye-mammography/releases/download/shared-models.v2/yolo8_m.pt) |
| <b> YOLOv8l | BI-RADS 4-5 <br> BI-RADS 1-2 | 0.805 <br> 0.765 | 0.08 <br> 0.361 | 0.871 <br> 0.823 | 0.847 | [checkpoint](https://github.com/ddobvyz/digitaleye-mammography/releases/download/shared-models.v2/yolo8_l.pt) |
| <b> YOLOv8x | BI-RADS 4-5 <br> BI-RADS 1-2 | 0.799 <br> 0.774 | 0.073 <br> 0.355 | 0.882 <br> 0.83 | 0.856 | [checkpoint](https://github.com/ddobvyz/digitaleye-mammography/releases/download/shared-models.v2/yolo8_x.pt) |
    
### **MMdetection Benchmarks**

Results and models are available in the below. Models will be automatically downloaded according to the selected model when running.

Models can also be downloaded from [the releases.](https://github.com/ddobvyz/digitaleye-mammography/releases/tag/shared-models.v1)
  
|   Model | Class | TPR | FPPI | AP | mAP | Checkpoints |
| :------: | :-----------: | :-------: | :-----------: | :---------------: | :---------: | :---------: |
| <b> Faster R-CNN  | BI-RADS 4-5 <br> BI-RADS 1-2 | 0.948 <br> 0.958 | 0.55 <br> 0.872 | 0.908 <br> 0.892 | 0.9  | [checkpoint](https://github.com/ddobvyz/digitaleye-mammography/releases/download/shared-models.v1/fasterrcnn.pth) |
| <b> DoubleHead R-CNN | BI-RADS 4-5 <br> BI-RADS 1-2 | 0.944 <br> 0.953 | 0.499 <br> 0.848 | 0.903 <br> 0.883 | 0.893 | [checkpoint](https://github.com/ddobvyz/digitaleye-mammography/releases/download/shared-models.v1/doublehead_rcnn.pth) |
| <b> Dynamic R-CNN | BI-RADS 4-5 <br> BI-RADS 1-2 | 0.943 <br> 0.958 | 0.519 <br> 0.791 | 0.898 <br> 0.882 | 0.89  | [checkpoint](https://github.com/ddobvyz/digitaleye-mammography/releases/download/shared-models.v1/dynamic_rcnn.pth) |
| <b> Cascade R-CNN | BI-RADS 4-5 <br> BI-RADS 1-2 | 0.948 <br> 0.96 | 0.516 <br> 0.865 | 0.9 <br> 0.884 | 0.892 | [checkpoint](https://github.com/ddobvyz/digitaleye-mammography/releases/download/shared-models.v1/cascade_rcnn.pth) |
| <b> YOLOv3 | BI-RADS 4-5 <br> BI-RADS 1-2 | 0.918 <br> 0.925 | 0.136 <br> 0.514 | 0.891 <br> 0.842 | 0.867 | [checkpoint](https://github.com/ddobvyz/digitaleye-mammography/releases/download/shared-models.v1/yolo_v3.pth) |
| <b> RetinaNET | BI-RADS 4-5 <br> BI-RADS 1-2 | 0.966 <br> 0.967 | 1.801 <br> 3.152 | 0.916 <br> 0.89 | 0.903  | [checkpoint](https://github.com/ddobvyz/digitaleye-mammography/releases/download/shared-models.v1/retina_net.pth) |
| <b> FCOS | BI-RADS 4-5 <br> BI-RADS 1-2 | 0.96 <br> 0.961 | 0.467 <br> 1.049 | 0.93 <br> 0.906 | 0.918 | [checkpoint](https://github.com/ddobvyz/digitaleye-mammography/releases/download/shared-models.v1/fcos.pth) |
| <b> VarifocalNET | BI-RADS 4-5 <br> BI-RADS 1-2 | 0.961 <br> 0.976 | 1.859 <br> 3.104 | 0.915 <br> 0.908 | 0.911 | [checkpoint](https://github.com/ddobvyz/digitaleye-mammography/releases/download/shared-models.v1/varifocal_net.pth) |
| <b> ATSS | BI-RADS 4-5 <br> BI-RADS 1-2 | 0.954 <br> 0.965 | 0.941 <br> 1.259 | 0.916 <br> 0.906 | 0.911 | [checkpoint](https://github.com/ddobvyz/digitaleye-mammography/releases/download/shared-models.v1/atss.pth) |
| <b> DETR | BI-RADS 4-5 <br> BI-RADS 1-2 | 0.941 <br> 0.969 | 0.493 <br> 1.025 | 0.913 <br> 0.899 | 0.906 | [checkpoint](https://github.com/ddobvyz/digitaleye-mammography/releases/download/shared-models.v1/detr.pth) |
| <b> DEDETR | BI-RADS 4-5 <br> BI-RADS 1-2 | 0.966 <br> 0.967 | 1.235 <br> 1.797 | 0.941 <br> 0.914 | 0.927 | [checkpoint](https://github.com/ddobvyz/digitaleye-mammography/releases/download/shared-models.v1/deformable_detr.pth) |
| <b> ENSEMBLE of BEST 3 MODELS <br> ATSS, DEDETR, FCOS | BI-RADS 4-5 <br> BI-RADS 1-2 | 0.958 <br> 0.968 | 0.575 <br> 1.020 | 0.937 <br> 0.922 | 0.929 | --- |
| <b> ENSEMBLE of ALL MODELS | BI-RADS 4-5 <br> BI-RADS 1-2 | 0.953 <br> 0.971 | 0.573 <br> 1.033 | 0.928 <br> 0.918 | 0.923 | --- |
    
<p align="left">
  <img src="docs/12_v4.png" width="400" />
  <img src="docs/45_v4.png" width="400" /> 
</p>   

## **Installation**

This application works torch 1.12.x and 10.2 <= cuda <= 11.6 between versions.

**Step 1.** Clone repo

```bash
git clone https://github.com/cbddobvyz/digitaleye-mammography.git
```

**Step 2.** Install requirements and mmcv-full version.
    
```bash
pip install -r requirements.txt
mim install mmcv_full==1.7.1
```
    
<b>Note: To perform install using Docker, please review the [docker readme file](docker/readme.md). </b>
    
**For using YOLO models please refer to [Ultralytics Docs](https://docs.ultralytics.com/).** 

## **Getting Started**

The parameters for running the toolkit in the terminal are provided below. 

```--model_enum:``` represents the user's model selection *(Required Parameter)*
```python
    {0: 'ATSS'
     1: 'CASCADE R-CNN'
     2: 'DEFORMABLE DETR'
     3: 'DETR'
     4: 'DOUBLEHEAD R-CNN'
     5: 'DYNAMIC R-CNN'
     6: 'FASTER R-CNN'
     7: 'FCOS'
     8: 'RETINANET'
     9: 'VARIFOCALNET'
     10: 'YOLOv3'
    }
```
    
```--img_path: ``` folder path for test images *(Required Parameter)*
    
```--device:``` running device *(Optional Parameter, Default: 'cpu')*
    
```--classify_mass:``` mass classification flag, if True it classifies mass as benign or malignant otherwise only mass detection performs *(Optional Parameter, Default: True)*
    
```--segment_breast: ``` breast segmentation for pre-processing *(Optional Parameter, Default: True)* [Download link for breast segmentation model](https://github.com/cbddobvyz/digitaleye-mammography/releases/download/shared-models.v1/ResUNet_breast.pth)
    
```--enable_ensemble: ``` applies ensemble *(Optional Parameter, Default: False)*

```--annotation_path: ``` annotation path for test images *(Optional Parameter, Default: None)* [Annotation file format from MMDetection](https://mmdetection.readthedocs.io/en/dev-3.x/advanced_guides/customize_dataset.html#an-example-of-customized-dataset)
    
```--nms_iou_threshold: ``` applies nms threshold to model results for post-processing (Optional Parameter, Default: 0.1)*
    
```--confidence_threshold: ``` applies confidence threshold to model results for post-processing (Optional Parameter, Default: 0.05)* 
    
```--ap_threshold: ``` IoU threshold for determining prediction as TP (True Positive) *(Optional Parameter, Default: 0.1)* 

### **Usage Examples**

Some of the usage examples are shown below.

* Only mass detection using DETR, RETINANET and YOLOv3 models with default parameters given test images.
  
```bash
python mass_inference.py --model_enum 3 8 10 --img_path test/img_paths/ --classify_mass False
```
* Mass classification and comparison against ground truth objects are performed on non-breast segmented images using ATSS, CASCADE R-CNN, and FASTER R-CNN models with various thresholds including non-maximum suppression (NMS), confidence, and average precision (AP).

```bash 
python mass_inference.py --model_enum 0 6 1 --img_path test/imgs_path/ --segment_breast False --annotation_path test/annot_path.txt --nms_iou_threshold 0.25 --confidence_threshold 0.5 --ap_threshold 0.1
```    

* Mass classification and model ensemble using DEFORMABLE DETR, DOUBLEHEAD R-CNN, DYNAMIC R-CNN, FCOS, VARIFOCAL NET models with various thresholds.

```bash 
python mass_inference.py --model_enum 2 4 5 7 9 --img_path test/imgs_path/ --enable_ensemble --nms_iou_threshold 0.1 --confidence_threshold 0.33 --ap_threshold 0.5
```

## Acknowledgements

This work was supported by Digital Transformation Office of the Presidency of Republic of Türkiye. We would like to thanks Republic of Türkiye Ministry of Health for sharing anonymized KETEM dataset and their valuable support.

## Citation

If you use this toolbox or benchmark in your research, please cite this paper.

```
@article{terzi2025digital,
  title={The digital eye for mammography: deep transfer learning and model ensemble based open-source toolkit for mass detection and classification},
  author={Terzi, Ramazan and Kılıç, Ahmet Enes and Karaahmetoğlu, Gökhan and Özdemir, Okan Bilge},
  journal={Signal, Image and Video Processing},
  volume={19},
  number={1},
  pages={170},
  year={2025},
  publisher={Springer}
}
```
## License
      
This project is released under the [GNU/GPLv3](https://github.com/ddobvyz/digitaleye-mammography/blob/main/LICENSE)
      
## Disclaimer

THERE IS NO WARRANTY FOR THE PROGRAM, TO THE EXTENT PERMITTED BY APPLICABLE LAW.  EXCEPT WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT HOLDERS AND/OR OTHER PARTIES PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM IS WITH YOU.  SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF ALL NECESSARY SERVICING, REPAIR OR CORRECTION.

THIS REPOSITORY DOES NOT PROVIDE MEDICAL ADVICE. THE INFORMATION, INCLUDING BUT NOT LIMITED TO, TEXT, GRAPHICS, IMAGES, AND OTHER MATERIAL CONTAINED ON THIS REPOSITORY ARE FOR INFORMATIONAL PURPOSES ONLY. NO MATERIAL ON THIS REPOSITORY IS INTENDED TO BE A SUBSTITUTE FOR PROFESSIONAL MEDICAL ADVICE, DIAGNOSIS, OR TREATMENT. ALWAYS SEEK THE ADVICE OF YOUR PHYSICIAN OR OTHER QUALIFIED HEALTH CARE PROVIDER WITH ANY QUESTIONS YOU MAY HAVE REGARDING A MEDICAL CONDITION OR TREATMENT, AND NEVER DISREGARD PROFESSIONAL MEDICAL ADVICE OR DELAY IN SEEKING IT BECAUSE OF SOMETHING YOU HAVE READ ON THIS REPOSITORY.
      
THE CONTENT OF REPOSITORY IS PROVIDED FOR INFORMATION PURPOSES ONLY. NO CLAIM IS MADE AS TO THE ACCURACY OR CURRENCY OF THE CONTENT ON THIS REPOSITORY AT ANY TIME. THE DIGITAL TRANSFORMATION OFFICE DOES NOT ACCEPT ANY LIABILITY TO ANY PERSON/INSTITUTION/ORGANIZATION FOR THE INFORMATION OR MODEL (OR THE USE OF SUCH INFORMATION OR MODEL) WHICH IS PROVIDED ON THIS REPOSITORY OR INCORPORATED INTO IT BY REFERENCE.
