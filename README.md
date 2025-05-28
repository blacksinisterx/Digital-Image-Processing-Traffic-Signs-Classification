# Traffic Sign Classification Using Classical Digital Image Processing Techniques

## Project Overview
This project implements a complete traffic sign classification system using classical digital image processing techniques without machine learning or pretrained models. The system analyzes traffic signs based on color, shape, and geometric features using only fundamental image processing operations.

## Dataset
- Selected 6-8 diverse traffic sign classes from the provided dataset
- Collected approximately 100 representative images per class (600-800 total images)
- Dataset structure follows the original format with organized class folders

## Implementation Pipeline

### 1. Data Preparation and Class Selection 
- Carefully selected diverse traffic sign classes for robust classification
- Organized dataset with proper folder structure and image referencing
- Ensured representative sampling from each class

### 2. Image Reading & Color Space Handling 
- Image loading using OpenCV (cv2.imread) or PIL (Image.open)
- Manual conversion to HSV color space
- All subsequent processing performed using NumPy arrays only

### 3. Image Preprocessing & Filtering 
Implemented the following filters from scratch using NumPy:
- **Mean Filter** (3×3 kernel)
- **Gaussian Filter** (with configurable standard deviation)
- **Median Filter** (noise reduction)
- **Adaptive Median Filter** (advanced noise handling)
- **Unsharp Masking/High-Boost Filtering** (edge enhancement)

### 4. Color Segmentation and Morphological Processing 
- **HSV Thresholding** for red and blue sign segmentation:
  - Red signs: Hue [0-15] or [165-180], Saturation ≥100, Value ≥80
  - Blue signs: Hue [100-130], Saturation ≥100, Value ≥80
- **Morphological Operations** (manually implemented):
  - Erosion and Dilation
  - Opening operations
  - Hole filling algorithms
  - Connected component filtering with area thresholds

### 5. Edge Detection and Region Extraction 
- **Manual Canny Edge Detection** implementation including:
  - Gradient computation using Sobel operators
  - Non-maximum suppression
  - Double thresholding and edge tracking
- Region of Interest (ROI) extraction with multi-blob handling logic

### 6. Geometric Normalization 
- Rotation angle calculation and orientation alignment
- Affine transformation construction and application using NumPy
- Uniform scaling to fixed dimensions (200×200 pixels)
- Optional perspective transformation capabilities

### 7. Feature Extraction 
Implemented comprehensive feature extraction:
- **Harris Corner Detection** (manual implementation)
- **Corner Count** calculation
- **Circularity**: C = 4π × Area / (Perimeter)²
- **Aspect Ratio** (width/height of bounding box)
- **Extent** (ratio of region area to bounding box area)
- **Average Hue** and color dominance analysis
- Feature normalization and transformation

### 8. Rule-Based Classification 
- Developed interpretable if-else classification rules
- Combined color and shape-based logic for robust classification
- Handles edge cases and visually similar signs (e.g., Stop vs. Yield)
- Efficient decision logic covering all 6-8 selected classes

### 9. Evaluation & Metrics 
Comprehensive evaluation system:
- Ground-truth comparison against Train.csv
- **results.csv**: filename, ground_truth, predicted, correct columns
- **metrics.txt**: Overall accuracy and class-wise precision, recall, accuracy
- **confusion_matrix.png**: Matplotlib-generated heatmap visualization

## Project Structure
```
i211234.zip
├── code/
│   ├── main.py                    # Main pipeline implementation
│   ├── preprocessing.py           # Filtering and preprocessing functions
│   ├── segmentation.py           # Color segmentation and morphological ops
│   ├── edge_detection.py         # Canny edge detector implementation
│   ├── feature_extraction.py     # Feature calculation functions
│   ├── classification.py         # Rule-based classifier
│   └── evaluation.py            # Metrics and evaluation functions
├── results.csv                   # Prediction results
├── metrics.txt                   # Performance metrics
└── confusion_matrix.png          # Confusion matrix visualization
```

## Key Features
- **100% Classical Approach**: No machine learning or pretrained models
- **Manual Implementation**: All algorithms implemented from scratch using NumPy
- **Robust Feature Set**: Multiple geometric and color-based features
- **Interpretable Rules**: Clear, logical classification rules
- **Comprehensive Evaluation**: Detailed performance analysis with visualizations

## Technical Highlights
- Advanced morphological processing with custom kernels
- Multi-stage edge detection with parameter optimization
- Geometric normalization handling various sign orientations
- Feature engineering optimized for traffic sign characteristics
- Rule-based system with hierarchical decision logic

## Results
The system successfully classifies traffic signs across 6-8 classes using only classical image processing techniques, demonstrating the power of fundamental computer vision operations when properly combined and optimized.

## Team Members
[Add your team member names and IDs here]

---
*This project demonstrates mastery of classical digital image processing techniques and their application to real-world computer vision problems.*
