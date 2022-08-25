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

There is a class imbalance of images with some classes with as low as 270 images. This may cause problems of overfitting because most machine learning algorithms assume that classes are balanced. With class imbalance, machine learning algorithms tend to be more biased towards the majority class and may misclassify minority classes. 

![](https://i.imgur.com/b6G9h3U.png)

Furthermore, if we examine pixel dimensions we that most images are around 30x30. Thus, we will reshape all images in the dataset to 30x30 because the machine learning algorithms we are using for image classification require all input images to have the same pixel dimensions.


Image Preprocessing
---
We have preprocessed the original dataset into 4 different datasets: original, grayscale, cropped, and cropped+grayscale. 

We preprocessed the original dataset in order to improve computational efficiency and test if we could improve accuracy.



Random Forest Classifier (RF)
---
Random forest is a supervised machine learning algorithm commonly used in classification or regression problems. It works by creating a large number of individual decision trees that each predict an output. The final prediction is the majority of predictions that were made. 

**Steps to run RF:** (same as SVM)
1. Read in training data (convert to grayscale, cropped, and cropped+grayscale while reading in training data)
2. Shuffle data to prevent overfitting
3. Read in test data (same process as reading in train data)
4. Normalize pixel values to increase efficiency
5. Convert all data arrays to 2d (because sklearn inputs need to be 2d arrays)
6. Fit and predict model

**RF Accuracy Results:**

| Original | Grayscale | Cropped | Cropped+Grayscale    |
| -------- | --------- | ------- | --- |
| 78.02% | 76.44% | 81.78% | 80.68% |




Support Vector Machine (SVM)
---
Support Vector Machine (SVM) is a supervised machine learning model that can be used for regression and classification problems but is mostly used for classification. SVM works by finding the optimal hyperplane in an n-dimensional space (where n is the number of features) that separates the classes. In this project, we have 43 classes of traffic signs so SVM will find the optimal hyperplane in a 43 dimensional space. The optimal hyperplane is the one with the maximum margin. This can be understood with an example of two classes where the maximum margin is the maximum distance between the datapoints of the two classes. For a more detailed explanation with visuals and a brief mathematical introduction click [here](https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47).

**Steps to Run SVM:** (same as RF)
1. Read in training data (convert to grayscale, cropped, and cropped+grayscale while reading in training data)
2. Shuffle data to prevent overfitting
3. Read in test data (same process as reading in train data)
4. Normalize pixel values to increase efficiency
5. Convert all data arrays to 2d (because sklearn inputs need to be 2d arrays)
6. Fit and predict model

**SVM Accuracy Results:**

| Original | Grayscale | Cropped | Cropped+Grayscale    |
| -------- | --------- | ------- | --- |
| 82.26% | 82.61% | 73.07% | 73.96% |

As we can see for SVM we get lower accuracies for the cropped and cropped+grayscale dataset when we would expect the opposite. This is likely due to some programming error when preprocessing the cropped images. We used the exact same method that was used to crop images for the RF model. However, we were unable to find where we went wrong in the process. One possible rememdy to get an expected higher accuracy for these datasets is to import the dataset from the RF notebook. However, we were short on time to test this since training the models took a substantial amount of time.

Convolutional Neural Networks (CNN)
---
CNNs are one of the most commonly used models for image classification because of their high accuracies, scalability to high dimensions, and lower data preprocessing demands. Therefore, we will be using the original dataset in our CNN model. 

A brief explanation for CNN does not do it justice so click [here](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1) for a detailed explanation.

Now we will explain some terms that show up in our CNN architecture below:

*Convolutional layer (Conv2D):* type of neuron layer that has filters/kernels which outputs feature scores

*LeakyReLU vs ReLU:* 
![](https://i.imgur.com/RMsEGy8.png)
ReLU is an activation function that is commonly used to remedy the [vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem). LeakyReLu is chosen for this model because some neurons can become stuck since the negative values of a ReLU function are 0.

*[Max pooling](https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/):* pooling operation that calculates maximum for each patch of the feature map as opposed to average pooling. Max pooling has been found to work better than average pooling for computer vision tasks.

*[Batch normalization](https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/):* when using deep neural networks, inputs from prior layers can change after weight updates. Thus, batch normalization is used to standardize the inputs at each layer. This improves training efficiency and can help prevent some overfitting.

*[Dropout](https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/):* ignores a certain percentage of nodes to reduce overfitting. Common practice is to have a dropout rate of 20%-50%.

*Flatten:* output is flattened into a vector that will then feed into the dense layer

*[Dense](https://machinelearningmastery.com/how-to-configure-the-number-of-layers-and-nodes-in-a-neural-network/):* layer of neurons where each neuron receives input from all the neurons of previous layer. 512 nodes are reduced to 43 nodes because there are 43 classes.


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

**Accuracy: 95.87%**


Conclusion
---
For image classification problems, neural networks are the best models to use. However, out CNN model is still insufficient for practical use in the real world. One problem to pay attention to would be the class imbalance. There are a couple remedies to class imbalance but unfortunately we were unable to employ them in this project. Articles on how to handle class imbalance will be in the **Useful Links** section.

Useful Links
---
- class imbalance: https://towardsdatascience.com/class-imbalance-a-classification-headache-1939297ff4a4
- https://machinelearningmastery.com/what-is-imbalanced-classification/
- https://www.linkedin.com/pulse/some-tricks-handling-imbalanced-dataset-image-m-farhan-tandia
- https://towardsdatascience.com/4-ways-to-improve-class-imbalance-for-image-data-9adec8f390f1
- https://machinelearningmastery.com/framework-for-imbalanced-classification-projects/
