import numpy as np
import matplotlib.pyplot as plt
import cv2 
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns  # Import seaborn
from cnn_Model import create_Model
import os
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# Load dataset
path = './dataset/color'
train_ds, test_ds = keras.utils.image_dataset_from_directory(
    path,
    image_size=(224, 224),
    batch_size=32,
    seed=123,
    validation_split=.2,
    subset='both'
)

##fetch classes from the dataset
classes = train_ds.class_names
test_classes = test_ds.class_names

# Create model
model = create_Model()
#compile the created model 
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics='accuracy'
)

model.summary()

# Load or train the model
if os.path.exists('plant_detection_model.h5'):
    model.load_weights('plant_detection_model.h5')
else:
    history = model.fit(train_ds, epochs=20)
    model.save_weights('plant_detection_model.h5')
    accuracy = history.history['accuracy']
    loss = history.history['loss']
    epochs = range(1, 21)
    plt.plot(epochs, accuracy, label='Accuracy')
    plt.plot(epochs, loss, label='Loss')
    plt.legend()
    plt.show()

# Visualize some test images with actual and predicted labels
plt.figure(figsize=(18, 18))
for images, labels in test_ds.take(1):
    for i in range(1, 10):
        plt.subplot(3, 3, i)
        plt.imshow(images[i].numpy().astype('uint32'))
        plt.axis('off')
        actual = classes[labels[i]]
        predict = classes[np.argmax(model.predict(tf.expand_dims(images[i], 0)))]
        plt.title(f"actual : {actual}  \n predicted : {predict}")

# Save a figure with test image predictions
figure_filename = 'result_figure.png'
figure_filepath = os.path.join('./results/', figure_filename)
plt.savefig(figure_filepath)
plt.close()

# Define a function to extract the group from a class name
def extract_group(class_name):
    return class_name.split('___')[0]

# Predictions on the test dataset using the trained model
test_predictions = model.predict(test_ds)

# Extract actual labels from the test dataset
test_labels = np.concatenate([y for x, y in test_ds], axis=0)

# Create a dictionary mapping each class to its respective group
class_groups = {class_name: extract_group(class_name) for class_name in test_classes}

# Extract predicted classes
predicted_classes = np.argmax(test_predictions, axis=1)

# Define classes to exclude from the analysis
exclude_classes = ['Grape', 'Corn_(maize)', 'Orange']

# Get unique groups excluding specified classes
unique_groups = np.unique([group for group in list(class_groups.values()) if group not in exclude_classes])

# Iterate through each unique group for analysis
for target_group in unique_groups:
    # Filter data for the current group
    group_indices = [idx for idx, class_name in enumerate(test_classes) if class_groups[class_name] == target_group]
    group_test_labels = test_labels[group_indices]
    group_predicted_classes = predicted_classes[group_indices]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

        # Create the confusion matrix
        conf_matrix = confusion_matrix(group_test_labels, group_predicted_classes)

        print(f"Confusion Matrix - {target_group}:")
        print(conf_matrix)

        print(f"\nClassification Report - {target_group}:")
        # Print a classification report with zero_division='warn'
        print(classification_report(group_test_labels, group_predicted_classes, zero_division='warn'))

        # Plot the confusion matrix using seaborn
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
        plt.ylabel('Prediction', fontsize=23)
        plt.xlabel('Actual', fontsize=23)
        plt.title(f'Confusion Matrix - {target_group}', fontsize=17)
        plt.show()
