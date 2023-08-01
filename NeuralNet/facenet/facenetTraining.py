from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Flatten, Dense
from sklearn.preprocessing import LabelBinarizer
from keras_vggface.vggface import VGGFace
from keras.preprocessing.image import ImageDataGenerator
from keras_facenet import FaceNet
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import os
import cv2
import glob

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load the pre-trained FaceNet model
model = FaceNet()

# Define the path to the directory containing the images
path = "/home/stanley/Documents/facerecognition/FR"

labels = ['daniel', 'jack', 'jinkuan', 'raj', 'stan', 'weimin']

# Load the images and labels into arrays
images = []
image_labels = []
for label in labels:
    image_paths = [os.path.join(path, label, f) for f in os.listdir(
        os.path.join(path, label)) if f.endswith('.jpg')]
    for image_path in image_paths:
        img = cv2.imread(image_path)
        # Resize the image to 160 x 160 pixels as expected by FaceNet
        img = cv2.resize(img, (160, 160))
        # Convert the image to RGB format as expected by FaceNet
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        image_labels.append(label)

# Convert the image and label arrays to NumPy arrays
images = np.array(images)
image_labels = np.array(image_labels)

# Get the embeddings of the images using the FaceNet model
embeddings = model.embeddings(images)

# Normalize the embeddings
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# Split the data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(
    embeddings, image_labels, test_size=0.2)

label_binarizer = LabelBinarizer()
train_labels_one_hot = label_binarizer.fit_transform(train_labels)
val_labels_one_hot = label_binarizer.transform(val_labels)

# Define a new model that takes the FaceNet embeddings as input
input_layer = Input(shape=(train_images.shape[1],))
x = Dense(512, activation='relu')(input_layer)
output_layer = Dense(len(labels), activation='softmax')(x)
new_model = Model(inputs=input_layer, outputs=output_layer)

# Compile the new model
new_model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

# Train the new model
history = new_model.fit(train_images, train_labels_one_hot,
                        epochs=500, validation_data=(val_images, val_labels_one_hot))

# Save the model
new_model.save('my_facenet_model2.h5')

# Evaluate the new model
loss, accuracy = new_model.evaluate(val_images, val_labels_one_hot)
print('Validation loss:', loss)
print('Validation accuracy:', accuracy)
