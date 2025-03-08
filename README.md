# Plant Leaf Disease Detection Using Machine Learning

This project implements a machine learning model to detect diseases in plant leaves using image classification techniques. It aims to enhance agricultural productivity by providing early and accurate disease detection.

## Project Overview

The main goal of this project is to accurately identify diseases in plants by analyzing leaf images.

- **Dataset:** The model was trained on a dataset containing images from 32 distinct diseased and healthy plants.
- **Data Split:** The dataset was split into a 75:25 ratio, with 75% for training and 25% for validation.
- **Model:** A Convolutional Neural Network (CNN) was used as the primary model.
- **User Interface:** A user-friendly interface was developed using Streamlit, allowing users to upload leaf images for disease prediction.

## Architecture

The architecture of the model is based on a Convolutional Neural Network (CNN), which is highly effective for image classification tasks.

1.  **Input Layer:** Accepts the input image data.
2.  **Convolutional Layers:** Extract features like edges, textures, and patterns from the images using filters.
3.  **Activation Layer (ReLU):** Introduces non-linearity to enable the model to learn complex patterns.
4.  **Pooling Layers (Max/Average):** Reduce the spatial dimensions of feature maps, lowering computational complexity.
5.  **Fully Connected Layers:** Classify the images into one of the 32 disease or healthy classes.
6.  **Dropout Layers:** Prevent overfitting by randomly deactivating neurons during training.
7.  **Output Layer:** Produces the final disease prediction.

## Technologies Used

-   **TensorFlow:** Machine learning and deep learning framework.
-   **Keras:** Deep learning API within TensorFlow for image preprocessing.
-   **Convolutional Neural Networks (CNNs):** Image classification model.
-   **Streamlit:** User interface framework.
-   **Matplotlib:** Data visualization.
-   **Pandas:** Data manipulation and analysis.
-   **Seaborn:** Advanced data visualization.

## Concepts

-   **TensorFlow:** Framework for building, training, and deploying neural networks.
-   **Keras:** Deep learning API for image preprocessing and model building.
-   **CNN (Convolutional Neural Network):** Neural network architecture for image analysis.
-   **Epoch:** One complete pass over the training dataset.
-   **Batch Size:** Number of samples processed before updating the model.
-   **Fully Connected Layer:** Layers that connect all neurons from the previous layer to every neuron in the current layer.
-   **ReLU (Rectified Linear Unit):** Activation function that introduces non-linearity.
-   **Pooling Layer:** Reduces spatial dimensions of feature maps.
-   **Dropout Layer:** Prevents overfitting by randomly deactivating neurons.
-   **Validation:** Process of evaluating model performance on unseen data.
-   **Metrics:** Evaluation measures like accuracy, precision, and recall.
-   **Data Augmentation:** Techniques to artificially increase the size of the dataset.
