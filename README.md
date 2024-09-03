# **YOLOv8 Collection Pruning**

This repository provides tools and scripts to perform pruning on YOLOv8 models to reduce their size and improve inference efficiency without significantly compromising accuracy. Pruning is an essential step in optimizing deep learning models for deployment in resource-constrained environments like mobile devices or edge computing.

## **Features**

- **Model Pruning**: Automatically prune YOLOv8 models to reduce the number of parameters and computations.
- **Customizable Pruning Ratios**: Set different pruning ratios for different layers or modules of the model.
- **Model Evaluation**: Evaluate the pruned model on benchmark datasets to verify performance metrics like mAP (mean Average Precision), inference speed, and model size.
- **Deployment Ready**: Export pruned models to formats that are ready for deployment in production environments.

## **Prerequisites**

Ensure you have the following dependencies installed:

- Python 3.7+
- PyTorch 1.8.0+
- torchvision
- numpy
- OpenCV
- YOLOv8 dependencies (if not already installed):
