# Plant Leaf Disease Detection
This repository focuses on detecting and classifying plant leaf diseases using advanced deep learning models. By leveraging Convolutional Neural Networks (CNN) and ResNet architectures, we aim to provide accurate and reliable disease identification. A straightforward web application is included, enabling users to upload an image of a plant leaf and receive real-time predictions.

## Table of Contents
- Introduction
- Key Features
- Dataset
- Models
- Project Structure
- Installation & Setup
- Running the Web App
- Results & Performance
- Future Work
- Contributing
- License

## Introduction
Plant diseases have a significant impact on crop quality and yield. Early and accurate detection is crucial for timely interventions. This project uses state-of-the-art deep learning techniques to automatically recognize diseases from leaf images, aiding farmers, researchers, and agronomists in making informed decisions.

## Key Features
Automatic Feature Extraction: CNN and ResNet models learn complex features (color patterns, spots, lesions) directly from images.
High Accuracy & Robustness: Residual connections in ResNet enable deeper networks, improving the model's ability to handle subtle differences between healthy and diseased leaves.
User-Friendly Web Interface: A simple web app lets users upload images and get instant disease predictions, making the tool accessible to non-technical stakeholders.

## Dataset
Dataset Paths: plant_images_train_test

The dataset directory contains training and testing images of various plant leaves. Each image is labeled either as healthy or corresponding to a particular disease. Ensure the dataset is structured into appropriate directories (e.g., train, test) for the models to ingest.

## Models
- Convolutional Neural Networks (CNN):
A baseline CNN architecture is used for initial disease classification, extracting features through convolutional and pooling layers.

- ResNet:
ResNet leverages residual blocks, allowing the training of much deeper networks without vanishing gradient issues. This typically results in improved accuracy and robustness compared to a basic CNN.

## Project Structure
- plant_images_train_test/
  - train/
  - test/
- models
  - cnn_model.h5
  - resnet_model.h5
  - ... (checkpoint, logs)
- notebooks
  - ads_599_leaf_disease_detection_Team_2.ipynb
- requirements.txt
- README.md

## Installation and Set Up
- Clone Repository:
  * git clone https://github.com/kaustav8319/ads-599-team-2.git
  * cd plant-leaf-disease-detection

- Create a virtual environment (On-MAC recommended):
   * python3 -m venv myenv
   * source myenv/bin/activate
-  On Windows: myenv\Scripts\activate
- Install dependencies: 
  * pip install --upgrade pip
  * pip install -r requirements.txt
 
## Results & Performance
### Model Accuracy:
The models (CNN and ResNet) achieve high accuracy on the test set. Refer to the evaluation.ipynb notebook for confusion matrices, classification reports, and detailed performance metrics.

### Model Complexity:
While the CNN model is simpler and runs faster, the ResNet model generally provides better accuracy but requires more computational resources.

## Future Work
### Enhanced Architectures:
Explore other architectures like EfficientNet or Vision Transformers for potentially better performance.

### Data Augmentation & Expansion:
Incorporate more plant species, disease types, and environmental conditions to improve model generalization.

### On-Device Inference:
Optimize models for mobile or embedded devices using frameworks like TensorFlow Lite, enabling field-level diagnostics.

## Contributing
Contributions are welcome! Feel free to open issues, submit pull requests, or suggest improvements.

## License
This project is licensed under the MIT License. You are free to use, modify, and distribute the code as needed.

