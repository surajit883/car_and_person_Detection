
# Person and Car Detection using YOLOv8

## Project Overview
This project focuses on detecting persons and cars in images using the YOLOv8 object detection model. The dataset provided includes images and corresponding annotations in JSON format. The goal was to convert the annotations to YOLOv8-compatible label files, split the data into training, validation, and test sets, and then train a model that accurately detects objects in new, unseen images.

## Why YOLOv8?
I chose YOLOv8 for the following reasons:
- **State-of-the-art object detection**: YOLOv8 is known for its balance between speed and accuracy, making it ideal for real-time object detection tasks.
- **Flexibility and ease of use**: YOLOv8 supports custom training, pre-trained weights, and easy integration into existing projects.
- **Efficiency**: YOLOv8 offers high performance, allowing us to detect multiple objects in an image with great accuracy, even in real-time applications.

## Model Name
**YOLOv8** (You Only Look Once Version 8)

## Links to Dataset and Framework
- **Dataset**: https://evp-ml-data.s3.us-east-2.amazonaws.com/ml-interview/openimages-personcar/trainval.tar.gz

## Explain About the Model
YOLOv8 is a state-of-the-art object detection model designed for real-time applications. It balances speed and accuracy, making it ideal for detecting multiple objects in images or video streams. The model processes images in a single pass, predicting bounding boxes and class probabilities directly, allowing for fast inference times.

## Primary Analysis
The dataset consisted of images and their corresponding annotations in JSON format. After converting the JSON annotations to the YOLO format, the data was split into training, validation, and test sets. The model was trained using the training dataset, and predictions were made on the test dataset.

## Assumptions
- The dataset is representative of the real-world scenarios where the model will be deployed.
- The annotations in the JSON files are accurate and complete.
- The YOLOv8 model is suitable for detecting persons and cars within the provided images.

## Inference
During the inference phase, the model processes unseen images to predict the presence and location of persons and cars. The model's output includes bounding boxes and class labels for the detected objects.

## False Positives
Some false pos

## Data Preparation
The provided dataset consisted of images and their corresponding annotation files in JSON format. The following steps were taken to prepare the data:
1. **Converted JSON to YOLO format labels**: The JSON annotation files were converted to the YOLO label format (text files) to be compatible with YOLOv8.
2. **Split dataset**: The images and their corresponding label files were divided into three subsets:
   - **Train set**: Used for training the model.
   - **Validation set**: Used for validating the model during training to prevent overfitting.
   - **Test set**: Used for evaluating the model's performance on unseen data.

## Folder Structure

```bash
📂 person-car-detection
├── 📂 data
│   ├── 📂 train
│   │   ├── 📂 images      
│   │   └── 📂 labels      
│   ├── 📂 val
│   │   ├── 📂 images      
│   │   └── 📂 labels      
│   ├── 📂 test
│   │   ├── 📂 images      
│   └── 📂 raw
│       ├── 📂 images      
│       ├── 📂 annotations  
│       └── 📂 labels       
├── 📂 models
│   └── best.pt
    └── best.onnx  
├── 📂 scripts
|   ├── convert_jason_to_yolo.py    
│   ├── split_data.py
|   ├── pt_to_onnx.py
│   ├── train.py    
│   ├── predict.py  
│   ├── check_iou_accuracy_v2.py
├── requirements.txt       
└── README.md              

```

## Project Steps

### 1. **Install Dependencies**
Make sure to install all the required dependencies before running the project:
create a virtual env and then install required packages

```bash
pip install -r requirements.txt
```

### 2. **jason to yolo fromat**
Before train the modle we need to convert our data to yolo format 
```bash
python src/convert_jason_to_yolo.py
```
### 2. **split the datset and preapare data for model**
Before train the modle we need to split our dataset .to split dataset  
```bash
python src/split_data.py
```
### 2. **Train the YOLOv8 Model**
The model was trained on the training dataset using the `train.py` script. This script handles loading the training data, configuring the model, and training the YOLOv8 architecture. To run the training:
```bash
python src/train.py
```
The model weights are saved to the `models/` directory as `best.pt`.

### 3. **Predict on Test Images**
After the model is trained, predictions can be made on the test dataset using the `predict.py` script:
```bash
python src/predict.py
```
### 3. **For check model accurcy on predict data**
After the model is predicted the images check tha accuracy of images `check_iou_accuracy_v2.py` script:
```bash
python src/check_iou_accuracy_v2.py

This script loads the trained model and runs it on the test images, saving the detection results to the appropriate folder.

### 4. **Evaluate the Model**
Once the test predictions are generated, the model's performance is evaluated in terms of:
- **Precision**: How accurate the model's predictions are.
- **Recall**: How many of the actual objects the model correctly detects.
- **mAP (0.5)**: Mean Average Precision at 0.5 IoU threshold.
- **mAP (0.5:0.95)**: Mean Average Precision over multiple IoU thresholds, giving a better understanding of the model's performance across different IoU levels.

Run the following command to get the evaluation metrics:
```bash
yolo task=test model=models/best.pt data=path_to_your_data.yaml
```
Make sure your `data.yaml` includes the paths to your test images.

## Conclusion
The YOLOv8 model was chosen for its high performance in object detection tasks, especially given its balance of speed and accuracy. During training, it demonstrated excellent performance in detecting persons and cars. The following metrics were recorded:

Per-Class Accuracy:
Class 1: 65.88%
Class 0: 55.10%
Per-Class Precision:
Class 1: 71.01%
Class 0: 62.53%
Overall Accuracy: 58.94%

Further improvements could involve fine-tuning the model with more data or utilizing additional augmentation techniques to enhance detection accuracy, especially in more complex scenarios.

