<center><h1> DIGITAL EYE PROJECT </h1></center>

## Introduction

- **WHAT's IT?**

    This repository contains trained models for mass detection and classification for cancer detection on mammography images.
    
- **WHY USE IT?**

    The shared models were trained with the MMdetection tool with a large data set and shared as a domain transfer learning resource for researchers. Researchers can test datasets in their current and future studies and finetune the models they will develop with shared models.
    
- **Dataset Description**

  The study was carried out using 33816 scan mammography images of 8454 devices in KETEM centers affiliated to the Ministry of Health of the Republic of Turkey. The dataset contains labels BI-RADS 12, BI-RADS 4 and BI-RADS 5. The BI-RADS 12 label is classified as BENIGN, the BI-RADS 4 and the BI-RADS 5 label as MALIGN.

- **Model Benchmarkings**

  The models were developed using the [MMdetection](https://github.com/open-mmlab/mmdetection) platform, which is used as an object detection tool under the [OpenMMLab](https://github.com/open-mmlab) project. config files contain hyper parameter and model architectures of the developed models. Result metrics were obtained by evaluating the models with test data. Models can be accessed via checkpoints.
    
|   Model Type |   Model  |  Config  | Class | TPR | FPPI | AP | mAP | Checkpoints |
| :------: | :------: | :-----------: | :-----------: | :-------: | :-----------: | :---------------: | :---------: | :---------: |
|  | <b> Faster R-CNN | fasterrcnn_config | BI-RADS 4-5 <br> BI-RADS 1-2 | 0.948 <br> 0.958 | 0.55 <br> 0.872 | 0.908 <br> 0.892 | 0.9  | [checkpoint](https://github.com/ddobvyz/digitaleye-mammography/releases/download/shared-models.v1/fasterrcnn.pth) |
|  | <b> DoubleHead R-CNN | doubleheadrcnn_config | BI-RADS 4-5 <br> BI-RADS 1-2 | 0.944 <br> 0.953 | 0.499 <br> 0.848 | 0.903 <br> 0.883 | 0.893 | [checkpoint](https://github.com/ddobvyz/digitaleye-mammography/releases/download/shared-models.v1/doublehead_rcnn.pth) |
|  | <b> Dynamic R-CNN | dynamicrcnn_config | BI-RADS 4-5 <br> BI-RADS 1-2 | 0.943 <br> 0.958 | 0.519 <br> 0.791 | 0.898 <br> 0.882 | 0.89  | [checkpoint](https://github.com/ddobvyz/digitaleye-mammography/releases/download/shared-models.v1/dynamic_rcnn.pth) |
|  | <b> Cascade R-CNN | cascadercnn_config | BI-RADS 4-5 <br> BI-RADS 1-2 | 0.948 <br> 0.96 | 0.516 <br> 0.865 | 0.9 <br> 0.884 | 0.892 | [checkpoint](https://github.com/ddobvyz/digitaleye-mammography/releases/download/shared-models.v1/cascade_rcnn.pth) |
|  | <b> YOLOv3 | yolov3_config | BI-RADS 4-5 <br> BI-RADS 1-2 | 0.918 <br> 0.925 | 0.136 <br> 0.514 | 0.891 <br> 0.842 | 0.867 | [checkpoint](https://github.com/ddobvyz/digitaleye-mammography/releases/download/shared-models.v1/yolo_v3.pth) |
|  | <b> RetinaNET | retinanet_config | BI-RADS 4-5 <br> BI-RADS 1-2 | 0.966 <br> 0.967 | 1.801 <br> 3.152 | 0.916 <br> 0.89 | 0.903  | [checkpoint](https://github.com/ddobvyz/digitaleye-mammography/releases/download/shared-models.v1/retina_net.pth) |
|  | <b> FCOS | fcos_config.py | BI-RADS 4-5 <br> BI-RADS 1-2 | 0.96 <br> 0.961 | 0.467 <br> 1.049 | 0.93 <br> 0.906 | 0.918 | [checkpoint](https://github.com/ddobvyz/digitaleye-mammography/releases/download/shared-models.v1/fcos.pth) |
|  | <b> VarifocalNET | varifocalnet_config | BI-RADS 4-5 <br> BI-RADS 1-2 | 0.961 <br> 0.976 | 1.859 <br> 3.104 | 0.915 <br> 0.908 | 0.911 | [checkpoint](https://github.com/ddobvyz/digitaleye-mammography/releases/download/shared-models.v1/varifocal_net.pth) |
|  | <b> ATSS | atss_config | BI-RADS 4-5 <br> BI-RADS 1-2 | 0.954 <br> 0.965 | 0.941 <br> 1.259 | 0.916 <br> 0.906 | 0.911 | [checkpoint](https://github.com/ddobvyz/digitaleye-mammography/releases/download/shared-models.v1/atss.pth) |
|  | <b> DETR | detr_config | BI-RADS 4-5 <br> BI-RADS 1-2 | 0.941 <br> 0.969 | 0.493 <br> 1.025 | 0.913 <br> 0.899 | 0.906 | [checkpoint](https://github.com/ddobvyz/digitaleye-mammography/releases/download/shared-models.v1/detr.pth) |
|  | <b> DEDETR | dedetr_config | BI-RADS 4-5 <br> BI-RADS 1-2 | 0.966 <br> 0.967 | 1.235 <br> 1.797 | 0.941 <br> 0.914 | 0.927 | [checkpoint](https://github.com/ddobvyz/digitaleye-mammography/releases/download/shared-models.v1/deformable_detr.pth) |

## USAGE

### COMMAND LINE ARGUMENTS    
    
Bu çalışmada kullanılan parametreler aşağıda gösterilmektedir.

```--model_enum:``` Kullanıcının model seçimini ifade eder.. (<b>Required Parameter,</b> Hangi modellerin kullanılacağı kullanıcı tarafından belirtilmelidir.) <br>
    <b>    
    {'0': 'ATSS',
     '1': 'CASCADE R-CNN',
     '2': 'DEFORMABLE DETR',
     '3': 'DETR',
     '4': 'DOUBLEHEAD R-CNN',
     '5': 'DYNAMIC R-CNN',
     '6': 'FASTER R-CNN',
     '7': 'FCOS',
     '8': 'RETINANET',
     '9': 'VARIFOCALNET',
     '10': 'YOLOv3'
    }
    </b> 
    
```--img_path: ``` Verilen mamografi görüntüsü klasör yolunu ifade eder. (<b>Required Parameter,</b> Hangi görüntüler üzerinde işlem yapılacağı kullanıcı tarafından belirtilmelidir. <br>
```--device:``` Test işlemlerinin hangi donanım üzerinde yapılacağı seçilir. (<b>Optional Parameter,</b> Default <b> 'cuda:0' </b> olarak setlenmiştir.) <br>
```--classify_mass:``` Verilen mamografi görüntüsünde kitle tespiti veya kitle sınıflandırılması seçimini ifade eder. (<b>Optional Parameter,</b> Default <b>True</b> olarak setlenmiştir. Modellerin kitle tespitiyle birlikte sınıflandırmasını içermektedir.) <br>
```--segment_breast: ``` Verilen mamografi görüntüsünde meme dokusunun segmente edilme seçimini ifade eder. (<b>Optional Parameter,</b> KETEM eğitim ve test pipeline'ında kullanıldığı için Default <b>True</b> olarak setlenmiştir.) <br>
```--enable_ensemble: ``` Verilen mamografi görüntüsü için modellerin ürettiği sonuçlara ensemble uygulanıp uygulanamayacağını ifade eder. <b>Optional Parameter,</b> Default <b>False</b> olarak setlenmiştir. <br>

```--annotation_path: ``` Verilen mamografi görüntüsü için annotation yolunu ifade eder. (Optional) <b>Optional Parameter,</b> kullanıcı isterse ground truth bilgilerini içeren text yolunu vererek, sonuç metriklerini hesaplatabilir. Default <b>None</b> olarak setlenmiştir. <br>
```--nms_iou_threshold: ``` Model tahminlerine NMS uygulanması için gerekli IoU değerini ifade eder. <b>Optional Parameter,</b> Default <b>0.1</b> olarak setlenmiştir. <br>
```--confidence_threshold: ``` Model tahminlerine skor eliminasyonu için gerekli skor değerini ifade eder. <b>Optional Parameter,</b> Default <b>0.05</b> olarak setlenmiştir. <br>
```--ap_threshold: ``` Model tahminlerinin GT ile kesişimi sonucu TP olarak değerlendirilmesi için gerekli olan IoU değerini ifade eder. <b>Optional Parameter,</b> Default <b>0.1</b> olarak setlenmiştir. <br>

### USAGE SCENARIOS    

#### Predictions on Image

```python3 mass_inference.py --model_enum ${MODEL_NUMBERS} --img_path ${IMAGE_FOLDER_PATH}```

Script verilen argümanlar ile birlikte çalıştırıldığında Repo içerisinde bir  <a href=".">repo_path</a> /workdirs klasörü oluşturulur. 'workdirs' klasörü içerisinde results klasörü oluşturularak model tahmini sonuçları görsellenir ve kaydedilir.

#### Predictions on Image with Ground Truth Annotations

```python3 mass_inference.py --model_enum ${MODEL_NUMBERS} --img_path ${IMAGE_FOLDER_PATH} --annotation_path ${ANNOTATION_PATH}```

Script verilen argümanlar ile birlikte çalıştırıldığında Repo içerisinde bir  <a href=".">repo_path</a> /workdirs klasörü oluşturulur. 'workdirs' klasörü içerisinde results klasörü oluşturularak model tahmini sonuçları görsellenir ve ground truth bilgileri ile birlikte kaydedilir.

#### Classify Predictions
    
```python3 mass_inference.py --model_enum ${MODEL_NUMBERS} --img_path ${IMAGE_FOLDER_PATH} --classify_mass ${BOOLEAN}``` 

#### Apply Model Ensemble Strategy

```python3 mass_inference.py --model_enum ${MODEL_NUMBERS}--img_path ${IMAGE_FOLDER_PATH} --enable_ensemble ${BOOLEAN}```

Script verilen argümanlar ile birlikte çalıştırıldığında Repo içerisinde bir  <a href=".">repo_path</a> /workdirs klasörü oluşturulur. 'workdirs' klasörü içerisinde results klasörü oluşturularak model tahmini sonuçları ve modellerin tahminlerine ensemble stratejisi uygulanarak sonuçlar verilir, görsellenir ve kaydedilir.
    
#### Apply Breast ROI Segmentation

```python3 mass_inference.py --model_enum ${MODEL_NUMBERS} --img_path ${IMAGE_FOLDER_PATH} --segment_breast ${BOOLEAN}```    

Script verilen argümanlar ile birlikte çalıştırıldığında Repo içerisinde bir  <a href=".">repo_path</a> /workdirs klasörü oluşturulur. 'workdirs' klasörü içerisinde results klasörü oluşturulur. Görüntüler, meme segmentasyon modeline girdi olarak verilir. Segmente edilen görüntüler results içerisinde breast_segmentation klasörüne kaydedilir. Kaydedilen görüntüler modellere girdi olarak verilir, görsellenir ve kaydedilir.
    
#### Change Post Process Parameters

- Change NMS IoU Threshold

```python3 mass_inference.py --model_enum ${MODEL_NUMBERS} --img_path ${IMAGE_FOLDER_PATH} --nms_iou_threshold ${FLOAT}```

Script verilen argümanlar ile birlikte çalıştırıldığında model tahmini sonuçlarına verilen IoU eşik değerinde NMS uygulanır. Model sonuçları görsellenir ve kaydedilir.

- Change Confidence Threshold

```python3 mass_inference.py --model_enum ${MODEL_NUMBERS} --img_path ${IMAGE_FOLDER_PATH} --confidence_threshold ${FLOAT}```

Script verilen argümanlar ile birlikte çalıştırıldığında model tahmini sonuçlarına verilen confidence eşik değerinde eliminasyon yapılır. Model sonuçları görsellenir ve kaydedilir.

- Change AP IoU Threshold

```python3 mass_inference.py --model_enum ${MODEL_NUMBERS} --img_path ${IMAGE_FOLDER_PATH} --annotation_path ${ANNOTATION_PATH} --ap_threshold ${FLOAT}```

Script verilen argümanlar ile birlikte çalıştırıldığında model tahmini sonuçları metrikleri için verilen IoU eşik değeri kullanılır. Metrikler bu parametre ile hesaplanır.

## How to Cite
    
    Makale link eklenecek.
    
## Acknowledgements
    
    Bu çalışma CBDDO Mamografi Görüntülerinden Meme Kanseri Tespiti projesi adı altında geliştirilmiştir.
    
<p align="center">
    <img src="https://cbddo.gov.tr/assets/img/footerlogo_en.png", alt="logo">
</p>
