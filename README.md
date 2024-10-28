# Face-Dissimilarity-Detection-using-Siamese-Network
This project utilizes a Siamese Network to measure the dissimilarity between faces in an image dataset. A Siamese Network is a type of neural network architecture that learns to differentiate between two inputs by embedding them into a shared feature space and calculating the distance between their representations. This is particularly useful for tasks such as face verification, where the goal is to determine whether two images belong to the same person or not.

The model is trained using pairs of face images, where each pair is labeled as either similar (same person) or dissimilar (different person). By leveraging contrastive loss, the network learns to embed faces in a high-dimensional space where similar faces are closer together, and dissimilar faces are far apart.

## Key Features:
- Face Dissimilarity Detection: Measure the similarity or dissimilarity between two face images.
- Deep Embedding: Use deep learning to embed faces into a feature space that captures facial characteristics.
- Customizable: The network can be retrained with different face datasets for applications like face verification, identification, and facial recognition.
This project can be extended to real-world applications such as security systems, identity verification, and automated image tagging.


## Project Structure

- `src/dataset.py`: Dataset class for loading and transforming image data.
- `src/network.py`: Defines the Siamese neural network and contrastive loss function.
- `src/train.py`: Script for training the model.
- `src/test.py`: Script for testing the model.
- `src/utils.py`: Helper functions for visualization.
- `app.py`: Streamlit app for making predictions.

## How to Run

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
