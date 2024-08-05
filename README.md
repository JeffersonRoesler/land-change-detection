# land-change-detection

Land change Detection - Capstone Project - Deep Learning with Pytorch

Project Overview

This project aims to detect land cover changes using satellite imagery, leveraging deep learning techniques. 
The primary objective is to monitor land use changes for purposes such as urban planning, environmental conservation, and property tax assessment.

Approach

Model: A ResNet-based neural network is used for feature extraction and classification.
Dataset: EuroSat Dataset with 13-band satellite images is utilized, covering 10 classes including categories like Forest, Residential, Industrial, Highway, and Permanent crops.


Model Architecture

Feature Extraction:

A pre-trained ResNet 18 model is used, modified to accept 13-channel input.

Change Detection Mechanism:

Computes absolute differences between features of image pairs.

Classification:

Two fully connected layers are used for classification.

Training and Validation

Loss Function: Cross-entropy loss
Optimizer: Adam with a learning rate of 0.001
Epochs: 10
The model achieves a test accuracy of 97%, indicating robust performance.

Change Detection

The project successfully demonstrates change detection using a simple model and a ResNet-based model.

Simulation of Construction

A small rectangle simulating a construction was inserted, showcasing the model's ability to detect changes.

Future Work and Improvements

Integration with Google Maps API:

Use geographical coordinates for precise location-based analysis, enhancing real-world applicability.

Integration with Geographic Information Systems (GIS):

Combine with GIS for comprehensive spatial analysis.

Temporal Analysis:

Incorporate temporal analysis to track changes over time.
