<center><h1> SAYISAL GÖZ PROJESİ </h1></center>

## GİRİŞ

   Meme kanseri kadınlar arasında görülen en yaygın kanser türüdür. Bu repo; mammografi görüntülerinde meme kanserinin erken teşhisine yardımcı olması amacıyla kitle tespiti ve sınıflandırılma için geliştirilmiş modelleri içermektedir. Paylaşılan modeller MMdetection aracı ile eğitilmiştir ve araştırmacılar için transfer learning kaynağı olarak paylaşılmıştır. Araştırmacılar, mevcut ve gelecekteki çalışmalarında veri setlerini test edebilir ve paylaşılan modeller ile geliştirecekleri modellerde ince ayar yapabilirler. 

   Bu çalışmada, Türkiye Cumhuriyeti Sağlık Bakanlığına bağlı KETEM merkezlerinde bulunan 8454 cihaza ait 33816 tarama mamografi görüntüsü kullanılmıştır. Veri kümesi BI-RADS 12, BI-RADS 4 ve BI-RADS 5 etiketlerini içermektedir. BI-RADS 12 etiketi BENIGN, BI-RADS 4 ve BI-RADS 5 etiketi MALIGN olarak sınıflandırılır.

## MODEL KIYASLAMA

  Paylaşılan 11 farklı nesne tanıma modeli, [OpenMMLab](https://github.com/open-mmlab) projesi altında nesne tanıma aracı olan [MMdetection](https://github.com/open-mmlab/mmdetection) platformu ile geliştirilmiştir. Paylaşılan modellerin test seti üzerindeki değerlendirme metrikleri sınıf bazlı olarak aşağıdaki tabloda gösterilmektedir.   
  
    
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

## KULLANIM

### KOMUT SATIRI ARGÜMANLARI    
    
Kodun çalıştırılması için kullanılan parametreler aşağıda gösterilmektedir.

```--model_enum:``` Kullanıcının model seçimini ifade eder.. (<b>Required Parameter,</b> Kullanıcı tarafından hangi modelin/modellerin kullanılacağı belirtilmelidir.) <br>
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
    
```--img_path: ``` Modellerin test edilmesi için gerekli mamografi görüntülerini içeren klasör yolunu ifade eder. (<b>Required Parameter</b>)<br>
```--device:``` Modellerin hangi donanım üzerinde çalışacağı seçilir. (<b>Optional Parameter,</b> Default <b> 'cpu' </b> olarak setlenmiştir.) <br>
```--classify_mass:``` Mamografi görüntülerinde modellerin, kitle tespiti veya kitle sınıflandırılması seçimini ifade eder. (<b>Optional Parameter,</b> Default <b>True</b> olarak setlenmiştir.) <br>
```--segment_breast: ``` Mamografi görüntülerinde meme bölgesine odaklanılması için segmentasyon modelinin çalışmasını ifade eder. (<b>Optional Parameter,</b> Default <b>True</b> olarak setlenmiştir.) <br>
```--enable_ensemble: ``` Modellerin ürettiği sonuçlara ensemble uygulanıp uygulanamayacağını ifade eder. <b>Optional Parameter,</b> Default <b>False</b> olarak setlenmiştir. <br>

```--annotation_path: ``` Verilen mamografi görüntüleri için ground truth annotation yolunu ifade eder. (Optional) <b>Optional Parameter,</b> Default <b>None</b> olarak setlenmiştir. <br>
```--nms_iou_threshold: ``` Model çıktılarına NMS uygulanması için gerekli IoU değerini ifade eder. <b>Optional Parameter,</b> Default <b>0.1</b> olarak setlenmiştir. <br>
```--confidence_threshold: ``` Model çıktılarına skor eliminasyonu için gerekli skor değerini ifade eder. <b>Optional Parameter,</b> Default <b>0.05</b> olarak setlenmiştir. <br>
```--ap_threshold: ``` Model çıktılarının ground truth ile kesişim eşik değerinin TP olarak değerlendirilmesi için gerekli olan IoU değerini ifade eder. <b>Optional Parameter,</b> Default <b>0.1</b> olarak setlenmiştir. <br>

### KULLANIM ÖRNEKLERİ    

Mamografi görüntülerinizi içeren verisetlerinizi paylaştığımız modeller üzerinde test etmek için sunduğumuz kullanım senaryolarını aşağıdan inceleyebilirsiniz. 

* Model tahmini sonuçları görsellenir ve kaydedilir.

```python3 mass_inference.py --model_enum ${MODEL_NUMBERS} --img_path ${IMAGE_FOLDER_PATH}```

* Mamografi görüntüleri, meme segmentasyon modeli ile segmente edilir, nesne tespit modeline girdi olarak verilir, sonuçlar görsellenir ve kaydedilir.

```python3 mass_inference.py --model_enum ${MODEL_NUMBERS} --img_path ${IMAGE_FOLDER_PATH} --segment_breast ${BOOLEAN}```    

* Model tahmini sonuçları ground truth bilgileri ile birlikte görsellenir ve  kaydedilir.

```python3 mass_inference.py --model_enum ${MODEL_NUMBERS} --img_path ${IMAGE_FOLDER_PATH} --annotation_path ${ANNOTATION_PATH}```

* Kitle tespiti veya sınıflandırılması seçimi yapılması
    
```python3 mass_inference.py --model_enum ${MODEL_NUMBERS} --img_path ${IMAGE_FOLDER_PATH} --classify_mass ${BOOLEAN}``` 

* Modellerin tahmini ve ensemble sonuçları görsellenir ve kaydedilir.
    
```python3 mass_inference.py --model_enum ${MODEL_NUMBERS}--img_path ${IMAGE_FOLDER_PATH} --enable_ensemble ${BOOLEAN}```
  
* Model çıktılarına verilen IoU eşik değerinde NMS uygulanır. Model sonuçları görsellenir ve kaydedilir.

```python3 mass_inference.py --model_enum ${MODEL_NUMBERS} --img_path ${IMAGE_FOLDER_PATH} --nms_iou_threshold ${FLOAT}```

* Model çıktılarına verilen confidence eşik değerinde eliminasyon yapılır. Model sonuçları görsellenir ve kaydedilir.
    
```python3 mass_inference.py --model_enum ${MODEL_NUMBERS} --img_path ${IMAGE_FOLDER_PATH} --confidence_threshold ${FLOAT}```

* TP olarak kabul edilecek IoU eşik değeri seçilir. Model çıktı metrikleri bu parametre ile hesaplanır.

```python3 mass_inference.py --model_enum ${MODEL_NUMBERS} --img_path ${IMAGE_FOLDER_PATH} --annotation_path ${ANNOTATION_PATH} --ap_threshold ${FLOAT}```


## How to Cite
    
    Makale link eklenecek.
    
## Acknowledgements
    
    Bu çalışma CBDDO Mamografi Görüntülerinden Meme Kanseri Tespiti projesi adı altında geliştirilmiştir.
