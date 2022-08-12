---
title: 'GTSRB'
disqus: hackmd
---

German Traffic Sign Recognition Benchmark (GTSRB)
===
**Contributors: Alexandre Miller, Justin Yeh**
## Table of Contents

[TOC]

## Introduction

In this repository, we will be exploring the German Traffic Sign Recognition Benchmark (GTSRB) dataset which can be found [here](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).

The GTSRB dataset contains over 50,000 images of traffic signs and 43 classes (unique traffic signs).

We aim to explore classification techniques to find which ones are best suited for image classification problems on large real life datasets.

The three methods we experimented with are Random Forest Classifier (RF), Support Vector Machine (SVM), and Convlutional Neural Networks (CNN).

Instructions
---
Each file can be run independently in colab. To load the kaggle datasets, go to your kaggle account and click on the **Create New API Token** button. Afterwards, upload the *kaggle.json* file to the /content folder and run the included commands in each file. Click [here](https://www.kaggle.com/general/74235) if still experiencing difficulties.

Files
---
*Exploratory Data Analysis: GermanStreetSign.ipynb*
*RF: GTSRB_RandomForestClassifier.ipynb*
*SVM: GTSRB_SVM.ipynb*
*CNN: GTSRB_CNN.ipynb*


Exploratory Data Analysis
---
![](https://i.imgur.com/HWXalUa.png)

There is an class imbalance of images with some classes with as low as 270 images. This may cause problems of overfitting because most machine learning algorithms assume that classes are balanced. With class imbalance, machine learning algorithms tend to be more biased towards the majority class and may misclassify minority classes. 

Image Preprocessing
---



Random Forest Classifier (RF)
---
Random forest is a supervised machine learning algorithm commonly used in classification or regression problems. It works by creating a large number of individual decision trees that each predict an output. The final prediction is the majority of predictions that were made. 


Support Vector Machine (SVM)
---

Convolutional Neural Networks (CNN)
---

**Structure**
| Layer (type)       | Output Shape       | Param # |
|--------------------|--------------------|---------|
| Conv2D             | (None, 30, 30, 32) | 896     |
| LeakyReLU          | (None, 30, 30, 32) | 0       |
| Conv2D             | (None, 28, 28, 32) | 9248    |
| LeakyReLU          | (None, 28, 28, 32) | 0       |
| MaxPooling         | (None, 14, 14, 32) | 0       |
| BatchNormalization | (None, 14, 14, 32) | 128     |
| Conv2D             | (None, 12, 12, 50) | 14450   |
| LeakyReLU          | (None, 12, 12, 50) | 0       |
| Conv2D             | (None, 10, 10, 50) | 22550   |
| LeakyReLU          | (None, 10, 10, 50) | 0       |
| MaxPooling         | (None, 5, 5, 50)   | 0       |
| BatchNormalization | (None, 5, 5, 50)   | 200     |
| Dropout            | (None, 5, 5, 50)   | 0       |
| Flatten            | (None, 1250)       | 0       |
| Dense              | (None, 512)        | 640512  |
| LeakyReLU          | (None, 512)        | 0       |
| Dropout            | (None, 512)        | 0       |
| Dense              | (None, 43)         | 22059   |



Conclusion
---

- put in confusion matrix for CNN
- class imbalance: https://towardsdatascience.com/class-imbalance-a-classification-headache-1939297ff4a4
- https://machinelearningmastery.com/what-is-imbalanced-classification/
- https://www.linkedin.com/pulse/some-tricks-handling-imbalanced-dataset-image-m-farhan-tandia
- https://towardsdatascience.com/4-ways-to-improve-class-imbalance-for-image-data-9adec8f390f1
- https://machinelearningmastery.com/framework-for-imbalanced-classification-projects/

## Appendix and FAQ

:::info
**Find this document incomplete?** Leave a comment!
:::

###### tags: `Templates` `Documentation`
