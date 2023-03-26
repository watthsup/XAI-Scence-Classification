# Scene Classification with Explainable CNN using GradCAM

This repository contains the source code for training a CNN model for scene classification, with the added feature of explainability using GradCAM. The model is trained on a dataset with six scene categories: buildings, forest, mountain, glacier, street, and sea.

## Key Features
- CNN model trained on scene classification dataset with six categories
- Adaptation of GradCAM for explainability of model predictions
- Provides EDA notebook for preliminary dataset understanding
- K-fold cross-validation to improve model robustness
- Easily configurable hyperparameters through "config.ini" file

## CNN Models
The project uses various CNN models, including ResNet, and DenseNet121. These models are implemented using the timm library, which provides a wide range of pre-trained models for computer vision tasks. Each model is trained using k-fold cross-validation, which helps to improve the generalization and to validate the robustness of the model performance.

## What is Grad-CAM?
Grad-CAM (Gradient-weighted Class Activation Mapping) is a technique used to highlight which regions of an image contribute the most to a particular class prediction made by a CNN. It generates a heatmap that shows which pixels of an image were most important in making the classification decision, making it a useful tool for visualizing and understanding the inner workings of a CNN.

## Libraries and Dependencies
- Pytorch
- Albumentations
- Numpy
- Pandas
- Matplotlib
- pytorch_gradcam
- segmentation_models
- timm
- scikit-learn
If you interested in using this project and have the problem about environment set up. Please contact me directly :D

## How to Use
To use this project, Please follow the following steps.
1. Set up your python environment and ensure that all the dependencies are installed
2. Navigate to the config.ini file to modify the hyperparameters if needed. 
3. The different CNN models can be trained using the train.py script.
4. Run python train.py to train your models

After training the models, the GradCam&Inference.ipynb notebook can be used to make predictions on new images and visualize the results with GradCam extension.

## Conclusion&Result
The models were trained using k-fold cross-validation, and the results show that the best model achieved an accuracy of 93% on the test set.

![image](https://user-images.githubusercontent.com/121663706/227762645-f0452d07-a6cd-4e59-84ea-bc5ca7e1ae72.png)

As seen from the below prediction result, it can be seen that with GradCam, it provides visual explanations to indicate which parts of the input image are contributing to the prediction. 
This can be helpful to diagnose why the CNN yield the incorrect result for further improvement by identifying which regions of the image the CNN is attending to and potentially adjusting the model or input accordingly.

![image](https://user-images.githubusercontent.com/121663706/227762657-0627641a-f5cf-4ac1-ac46-02d480f83895.png)

