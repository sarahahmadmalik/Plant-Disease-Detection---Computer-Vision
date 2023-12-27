import numpy as np
import matplotlib.pyplot as plt
import cv2 
import tensorflow as tf
from tensorflow import keras
from cnn_Model import create_Model
import os

path = './dataset/color'
train_ds, test_ds = keras.utils.image_dataset_from_directory(
    path,
    image_size=(224,224),
    batch_size=32,
    seed=123,
    validation_split=.2,
    subset='both'
)

classes = train_ds.class_names

image_save_path = './results/'  
os.makedirs(image_save_path, exist_ok=True)

model = create_Model()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics='accuracy'
)

model.summary()

if os.path.exists('plant_detection_model.h5'):
    model.load_weights('plant_detection_model.h5')
else:
    history = model.fit(train_ds, epochs=20)
    model.save_weights('plant_detection_model.h5')

accuracy = history.history['accuracy']
loss = history.history['loss']
epochs = range(1, 21)

plt.plot(epochs, accuracy, label='Acuuracy')
plt.plot(epochs, loss, label='loss')
plt.legend()
plt.show()

model.evaluate(test_ds)

def img_to_pred(image):
    image = image.numpy()
    image = tf.expand_dims(image, 0)
    return image

plt.figure(figsize=(18, 18))
for images, labels in test_ds.take(1):  
    for i in range(1, 10):
        plt.subplot(3, 3, i)
        plt.imshow(images[i].numpy().astype('uint32'))
        plt.axis('off')
        actual = classes[labels[i]]
        prediction = classes[np.argmax(model.predict(img_to_pred(images[i])))]
        plt.title(f"actual : {actual}\npredicted : {prediction}")

        image_filename = f"actual_{actual}_predicted_{prediction}_image_{i}.png"
        image_filepath = os.path.join(image_save_path, image_filename)
        plt.savefig(image_filepath)
        plt.close()

print("Images saved in:", image_save_path)