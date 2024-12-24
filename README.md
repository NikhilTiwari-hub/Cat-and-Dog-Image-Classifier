# Cat-and-Dog-Image-Classifier
# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Set image dimensions and batch size
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32

# Create image generators with rescaling
train_image_gen = ImageDataGenerator(rescale=1./255, 
                                     rotation_range=40,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     shear_range=0.2,
                                     zoom_range=0.2,
                                     horizontal_flip=True,
                                     fill_mode='nearest')

validation_image_gen = ImageDataGenerator(rescale=1./255)
test_image_gen = ImageDataGenerator(rescale=1./255)

# Flow from directories for training, validation, and testing
train_data_gen = train_image_gen.flow_from_directory(
    'cats_and_dogs/train',
    batch_size=BATCH_SIZE,
    class_mode='binary',
    target_size=(IMG_HEIGHT, IMG_WIDTH)
)

validation_data_gen = validation_image_gen.flow_from_directory(
    'cats_and_dogs/validation',
    batch_size=BATCH_SIZE,
    class_mode='binary',
    target_size=(IMG_HEIGHT, IMG_WIDTH)
)

test_data_gen = test_image_gen.flow_from_directory(
    'cats_and_dogs/test',
    batch_size=BATCH_SIZE,
    class_mode=None,  # No labels for test set
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    shuffle=False  # Keep the order for predictions
)

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_data_gen,
    steps_per_epoch=train_data_gen.samples // BATCH_SIZE,
    epochs=10,
    validation_data=validation_data_gen,
    validation_steps=validation_data_gen.samples // BATCH_SIZE
)

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# Predict probabilities on the test data
predictions = model.predict(test_data_gen)
probabilities = predictions.flatten()

# Plot the test images with their predicted labels and probabilities
def plotImages(data_gen, probabilities):
    images = [data_gen[i][0][0] for i in range(5)]  # Get the first 5 images
    labels = ['Dog' if prob > 0.5 else 'Cat' for prob in probabilities[:5]]
    
    fig, axes = plt.subplots(1, 5, figsize=(15, 10))
    for ax, img, label, prob in zip(axes, images, labels, probabilities[:5]):
        ax.imshow(img)
        ax.set_title(f'{label}: {prob*100:.2f}%')
        ax.axis('off')
    plt.show()

plotImages(test_data_gen, probabilities)
