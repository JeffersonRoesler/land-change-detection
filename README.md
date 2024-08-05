# land-change-detection
Land change Detection - Capstone Project - Deep Learning with Pytorch

Project Overview
This project aims to detect land cover changes using satellite imagery, leveraging deep learning techniques. The primary objective is to monitor land use changes for purposes such as urban planning, environmental conservation, and property tax assessment.

Approach
Model: A ResNet-based neural network is used for feature extraction and classification.
Dataset: EuroSat Dataset with 13-band satellite images is utilized, covering 10 classes including categories like Forest, Residential, Industrial, Highway, and Permanent crops.
Project Requirements
Dataset Selection:

Choose a suitable image dataset for your project, such as the EuroSat Dataset.
Ensure a reasonable number of classes and a sufficient number of images per class. If the dataset is too large, consider using a subset.
Data Preprocessing:

Perform data preprocessing steps like resizing images, normalizing pixel values, and splitting the dataset into training, validation, and test sets.
Apply data augmentation techniques to enhance training data diversity.
Model Selection and Architecture:

Select a deep learning architecture like a convolutional neural network (CNN).
Define the architecture, including layers, activation functions, and regularization techniques.
Model Training:

Train the model using the training dataset.
Monitor training progress, including loss and accuracy, and consider early stopping to prevent overfitting.
Hyperparameter Tuning:

Experiment with hyperparameters (e.g., learning rate, batch size) to optimize performance.
Keep a record of hyperparameters and their impact on the model.
Evaluation:

Evaluate the model using the validation dataset, assessing performance through metrics like accuracy, precision, recall, and F1-score.
Visualize predictions and misclassifications.
Fine-Tuning and Iteration:

Adjust the model architecture or hyperparameters based on evaluation results.
Reiterate training and evaluation to achieve satisfactory performance.
Final Model Testing:

Test the final model on the held-out test dataset to assess generalization to unseen data.
Documentation and Reporting:

Create a report summarizing the dataset, model architecture, training process, evaluation results, and insights.
Include visualizations and explanations for clarity.
Presentation:

Prepare a presentation showcasing the project's key findings and outcomes.
Share experiences, challenges faced, and lessons learned.
Conclusion:

Summarize achievements and any potential future work or improvements to the model.
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

Results
Confusion Matrix
The confusion matrix demonstrates the performance across different classes, highlighting the model's accuracy and potential areas of misclassification.

Predictions
Visualizations of the model's predictions help illustrate its strengths and areas for improvement.

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
