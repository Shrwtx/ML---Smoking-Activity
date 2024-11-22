# YOLOv8-Based Smoking Activity Detection

## Project Overview
This project involves the development of a deep learning-based system for the detection of smoking activities in real-time, using the YOLOv8 (You Only Look Once) architecture. The model was trained to identify key smoking-related objects, including faces, cigarettes, and smoking gestures, achieving high precision and recall for each category. The system is designed for applications in public health surveillance, specifically for enforcing smoking bans in public spaces.

## Features
- **Real-Time Smoking Detection:** Detects smoking-related behaviors such as cigarette holding, face identification, and smoking actions.
- **High Precision & Recall:** Achieved 95.6% precision, 94.2% recall, and 96.9% mean Average Precision (mAP) at IoU=0.5.
- **Class-Wise Performance:** 
  - **Face Detection:** 98.9% precision, 97.9% recall.
  - **Cigarette Detection:** 93.5% precision, 87.5% recall.
  - **Smoking Gesture Detection:** 96.2% precision, 97.3% recall.
  
## Installation & Setup

### Prerequisites
To run the model, ensure the following dependencies are installed:
- Python 3.7+
- PyTorch 1.7+ (with CUDA support if running on GPU)
- YOLOv8 (Ultralytics)
- OpenCV
- numpy
- matplotlib

### Steps to Install
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/smoking-detection.git
   cd smoking-detection
   ```

2. **Create a Virtual Environment (Optional but Recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # For Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download YOLOv8 Pre-trained Weights (if applicable):**
   You can download the weights for YOLOv8 from the official [Ultralytics YOLOv8](https://docs.ultralytics.com/) repository or use the training script to retrain it on custom data.

## Dataset
The model was trained using the **Smoking Object Detection Dataset** (v3, 2024-06-12) sourced from [Roboflow](https://universe.roboflow.com/yeolduri/smoking-hvni4/dataset/3). The dataset includes images annotated for faces, cigarettes, and smoking gestures.

## Training
1. **Start Training:**
   If you wish to retrain the model on custom data, run:
   ```bash
   python train.py --data custom_dataset.yaml --weights yolov8.pt --epochs 25
   ```

2. **Custom Dataset Format:**
   The dataset should follow the YOLO annotation format. Ensure that the annotations are in text files where each line represents an object in the format:
   ```
   class_id x_center y_center width height
   ```

## Evaluation
The model has been evaluated based on the following metrics:
- **Precision:** 95.6% (high accuracy in predicting smoking-related instances).
- **Recall:** 94.2% (high ability to detect all smoking-related instances).
- **Mean Average Precision (mAP):** 96.9% at IoU=0.5.
- **Class-Wise Results:**
  - **Face Detection:** Precision = 98.9%, Recall = 97.9%
  - **Cigarette Detection:** Precision = 93.5%, Recall = 87.5%
  - **Smoking Gesture Detection:** Precision = 96.2%, Recall = 97.3%

### Confusion Matrix
The confusion matrix is available for further analysis, which helps visualize the model’s performance across different categories and highlight misclassifications.

## Results & Visualizations
- **F1-Score Curve:** Plots F1 score against confidence thresholds.
- **Precision-Recall Curve:** Evaluates the trade-off between precision and recall at various thresholds.
- **Loss and Metric Curves:** Tracks box loss, classification loss, and other key metrics during training.

### Sample Output
- **Face Detection:** Accurately identifies individuals involved in smoking activity.
- **Cigarette Detection:** Identifies cigarettes held by individuals.
- **Smoking Gestures Detection:** Recognizes smoking actions like lighting or inhaling.

## Testing on New Data
The model is designed to work on both images and video inputs. For video-based testing, the system processes frames in real-time to detect smoking behaviors.

### Example:
```bash
python detect.py --source video_path_or_image_path --weights yolov8.pt
```

## Conclusion
This YOLOv8-based smoking detection system demonstrates high accuracy and robustness in detecting smoking activities in diverse environments. It is suitable for real-time surveillance in public spaces, contributing to health monitoring and enforcement of smoking bans.

## Future Work
Future improvements could include:
- **Integration with IoT devices:** Combining with smoke detectors or air quality sensors for a more holistic solution.
- **Improving Model Generalization:** Extending the system's application to various lighting conditions and indoor settings.
- **Optimization for Low-Resource Devices:** Scaling the model for deployment on edge devices with limited computational resources.

## References
1. YOLOv8: Scalable Object Detection, [Ultralytics](https://docs.ultralytics.com/)
2. Smoking Object Detection Dataset v3, 2024-06-12, [Roboflow](https://universe.roboflow.com/yeolduri/smoking-hvni4/dataset/3)
3. Redmon, J., & Farhadi, A. (2018). YOLOv3: An Incremental Improvement. arXiv preprint arXiv:1804.02767.
4. Wang, C.-Y., Bochkovskiy, A., & Liao, H.-Y. M. (2020). YOLOv4: Optimal Speed and Accuracy of Object Detection. arXiv preprint arXiv:2004.10934.

