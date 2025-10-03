# **Road Vehicle Detection**

This project implements a road vehicle detection system using deep learning. The system detects vehicles like cars, buses, bikes, trucks, and bicycles in road scene images using YOLOv3, YOLOv5, YOLOv8, Faster R-CNN, and a custom CNN model. It also includes data preprocessing, bounding box conversion, visualization, and performance analysis pipelines.

### **Features**

Detects multiple vehicle classes from images.

Supports YOLOv3, YOLOv5, YOLOv8, Faster R-CNN, and custom CNN models.

Converts bounding boxes between YOLO, VOC, and COCO formats.

Visualizes detection results with bounding boxes and confidence scores.

Generates heatmaps, confidence distribution plots, and CSV summaries.

### **Datasets**

RSUD20K Bangladesh Road Scene Dataset

Road Vehicle Images Dataset

Color Palette Dataset

Note: Download the datasets and update the paths in the code accordingly.

### **Installation**

Clone the repository:

```git clone https://github.com/<your-username>/road-vehicle-detection.git
cd road-vehicle-detection```


Install dependencies:

```pip install -r requirements.txt```


Install additional packages (if needed):

```pip install kagglehub pybboxes opencv-python-headless ultralytics torch torchvision matplotlib seaborn```

### **Usage**

Data Preprocessing

```python src/data_loader.py```


Prepares CSV files for train, validation, and test sets. 

Train Custom CNN Model

```python src/train.py```


Run Detection with YOLO/Faster R-CNN

``` python src/inference.py --model yolov8 --image path/to/image.jpg ```


Visualize Results

```python src/visualize.py --input path/to/image.jpg```

### **Results**

Detection accuracy heatmaps and confidence score distributions are generated for evaluation.

Sample images with bounding boxes and confidence scores are saved in results/images/.

### **Project Structure**
```
road-vehicle-detection/
│── data/                 # datasets
│── notebooks/            # Colab/Jupyter experiments
│── src/                  # scripts
│    ├── data_loader.py
│    ├── train.py
│    ├── inference.py
│    ├── visualize.py
│── results/
│    ├── detections.csv
│    ├── images/
│── requirements.txt
│── README.md
```

### **Technologies Used**

Python, OpenCV, NumPy, Pandas, Matplotlib, Seaborn

PyTorch, TensorFlow/Keras

YOLOv3, YOLOv5, YOLOv8, Faster R-CNN

### **License**

This project is open-source and available under the MIT License
