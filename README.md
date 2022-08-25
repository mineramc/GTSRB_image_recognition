---
title: 'GTSRB'
disqus: hackmd
---

German Traffic Sign Recognition Benchmark (GTSRB)
===
**Contributors: Alexandre Miller, Justin Yeh**

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

![](https://i.imgur.com/b6G9h3U.png)

Furthermore, if we examine pixel dimensions we that most images are around 30x30. Thus, we will reshape all images in the dataset to 30x30 because the machine learning algorithms we are using for image classification require all input images to have the same pixel dimensions.


Image Preprocessing
---
We have preprocessed the original dataset into 4 different datasets: original, grayscale, cropped, and cropped+grayscale.



Random Forest Classifier (RF)
---
Random forest is a supervised machine learning algorithm commonly used in classification or regression problems. It works by creating a large number of individual decision trees that each predict an output. The final prediction is the majority of predictions that were made. 

Steps to run RF:
1. Read in training data (convert to grayscale, cropped, and cropped+grayscale while reading in training data)
2. Shuffle data to prevent overfitting
3. Read in test data (same process as reading in train data)
4. Normalize pixel values to increase efficiency
5. Convert all data arrays to 2d (because sklearn inputs need to be 2d arrays)
6. Fit and predict model

**RF Results:**

| Original | Grayscale | Cropped | Cropped+Grayscale    |
| -------- | --------- | ------- | --- |
| 78.02% | 76.44% | 81.78% | 80.68% |




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
