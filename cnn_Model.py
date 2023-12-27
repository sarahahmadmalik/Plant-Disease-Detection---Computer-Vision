import tensorflow as tf
from tensorflow import keras

def create_Model():
    # Creating a Sequential model
    model = keras.Sequential([
        # Rescale input pixel values to the range [0, 1]
        keras.layers.Rescaling(scale=1/255, input_shape=(224, 224, 3)),
        
        # Convolutional layers with ReLU activation, max pooling, and dropout
        # Explanation: Convolutional layers help capture spatial patterns in the image. 
        # ReLU activation introduces non-linearity. Max pooling reduces spatial dimensions.
        # Dropout helps prevent overfitting by randomly setting a fraction of input units to zero.
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Dropout(0.2),
        
        # Additional convolutional layers with similar patterns
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Dropout(0.2),
        
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Dropout(0.2),
        
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Dropout(0.2),
        
        # A deeper convolutional layer with increased filters
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.MaxPool2D((2, 2)),
        
        # Flatten the output for the fully connected layers
        # Explanation: Flatten transforms the multi-dimensional output into a flat vector.
        # This is necessary when transitioning from convolutional layers to fully connected layers.
        keras.layers.Flatten(),
        
        # Fully connected layers with ReLU activation
        # Explanation: These layers process the flattened features from the convolutional layers.
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        
        # Output layer with sigmoid activation for multi-label classification
        # Explanation: Sigmoid activation is used for multi-label classification tasks.
        # It outputs probabilities for each class independently.
        keras.layers.Dense(38, activation='sigmoid')
    ])

    return model
