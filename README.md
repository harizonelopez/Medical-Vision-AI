# 🫁 MedVision AI

> AI-powered chest X-ray analysis with explainable deep learning.

![Blue technology badge reading Python 3.12](https://img.shields.io/badge/Python-3.12-blue)
![Red technology badge reading PyTorch Deep Learning](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Blue technology badge reading Computer Vision](https://img.shields.io/badge/Computer-Vision-blue)
![Cyan technology badge reading Explainability GradCAM](https://img.shields.io/badge/Explainability-GradCAM-cyan)
![Green technology badge reading Frontend Streamlit](https://img.shields.io/badge/Frontend-Streamlit-green)

MedVision AI is an educational computer-vision project that demonstrates how a deep-learning model can be used to explore chest X-ray classification and visualize image regions that influence a prediction. It combines PyTorch inference with an interactive Streamlit interface and Grad-CAM explainability.

> **Important:** This is a research and educational prototype, not a medical device. Its output is not a diagnosis and must not be used to guide clinical decisions.

## Contents

- [Overview](#overview)
- [Features](#features)
- [How it works](#how-it-works)
- [Model and task](#model-and-task)
- [Dataset](#dataset)
- [Evaluation results](#evaluation-results)
- [Technology stack](#technology-stack)
- [Project limitations](#project-limitations)
- [Future work](#future-work)

## Overview

MedVision AI analyzes an uploaded chest X-ray and presents a pneumonia-class probability alongside an interactive Grad-CAM visualization. The dashboard also provides an adjustable decision threshold, model performance metrics, a downloadable research report, and a session history of analyses.

The project covers an end-to-end machine-learning workflow: dataset preparation and patient-level splitting, model training with transfer learning, held-out evaluation, and interactive inference. It is intended to make these components accessible for demonstration and exploration.

## Features

- **Image inference:** Run a binary classification model on an uploaded chest X-ray.
- **Probability display:** View the model's pneumonia-class probability.
- **Threshold control:** Adjust the decision threshold used to interpret the model score. Changing the threshold can alter the predicted class; it does not change the underlying probability or improve model calibration.
- **Grad-CAM visualization:** Explore a heatmap of image regions associated with the model's output. This visualization is an explanation aid, not proof that the model is attending to medically meaningful evidence.
- **Evaluation information:** Review model performance metrics and ROC-curve evaluation.
- **Research report:** Generate and download a report for an analysis.
- **Session history:** Review analyses made during the current app session.
- **Training considerations:** Use patient-level train/validation/test splits and weighted loss to address class imbalance.

## How it works

At a high level, the workflow is:

- Prepare the chest X-ray data and create patient-level training, validation, and test splits.
- Train a ResNet-18 binary classifier using PyTorch and transfer learning, with weighted loss to account for class imbalance.
- Evaluate the model on a held-out test set, including ROC-AUC, sensitivity, and specificity.
- Upload an image through the Streamlit dashboard to view a model score and Grad-CAM visualization.
- Optionally adjust the displayed decision threshold, review session history, and download a research report.


## Model and task

- **Architecture:** ResNet-18
- **Framework:** PyTorch
- **Approach:** Transfer learning
- **Task:** Binary classification, **No Finding** vs **Pneumonia**

The model's output is a learned score for this defined task. It should not be interpreted as a complete assessment of a patient's health or as a substitute for expert review.

## Dataset

The project uses the **NIH ChestX-ray14 Dataset**. The dataset is not included in this repository because of licensing and file-size considerations. Users must obtain and use the data according to its applicable terms.

Patient-level splitting is used to separate training, validation, and test data. This helps avoid putting images from the same patient in multiple splits, which could otherwise make evaluation overly optimistic.

## Evaluation results

Reported performance on the held-out test set:

|  Metric       |  Result      |
|---------------|--------------|
|  ROC-AUC      |  **0.7744**  |
|  Sensitivity  |  **76.3%**   |
|  Specificity  |  **65.1%**   |

These are project-reported results for the stated test set and evaluation setup. Performance may vary across datasets, populations, image acquisition conditions, and decision thresholds. Sensitivity and specificity are threshold-dependent; these metrics alone do not establish clinical utility or safety.

## Technology Stack

> MedVision AI is containerized using Docker, providing a consistent runtime environment with the application's dependencies and configuration.

**Core Technologies**
- **Python 3.12** - *Main programming language.*
- **PyTorch** - *Deep learning framework.*
- **Torchvision** - *Image processing and pretrained computer vision models.*
- **Streamlit** - *Interactive web application framework.*
- **NumPy** - *Numerical computing.*
- **Pandas** - *Data manipulation and analysis.*
- **Pillow** - *Image loading and processing.*
- **Grad-CAM** - *Visual explanations of model predictions.*
- **Docker** - *Containerization and environment consistency.*

## Project limitations

- The system is a prototype for education and research, not a validated clinical tool.
- The task is limited to the two classes **No Finding** and **Pneumonia**; it is not a comprehensive multi-condition assessment.
- Model predictions and Grad-CAM maps can be incorrect or misleading.
- Dataset-specific test results do not guarantee performance on external data or in clinical settings.
- No dataset is bundled with the repository.

## Future work

- *Explore Vision Transformers*
- *Add DICOM support and batch inference*
- *Improve confidence calibration and inference speed*
