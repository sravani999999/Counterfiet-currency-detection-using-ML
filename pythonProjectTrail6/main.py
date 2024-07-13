import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Resize the image to a fixed size (e.g., 224x224) if needed
    image = cv2.resize(image, (224, 224))

    image = cv2.resize(image, (224, 224))

    # Normalize pixel values to the range [0, 1]
    normalized_image = image / 255.0

    return normalized_image


# Define the dataset paths
train_dir = 'currency_dataset/train'
validation_dir = 'currency_dataset/validation'
test_dir = 'currency_dataset/test'

# Define image dimensions
image_size = (224, 224)
batch_size = 32

# Create data generators with data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

# Create a base MobileNetV2 model (you can choose other architectures as well)
base_model = MobileNetV2(weights='imagenet', include_top=False,
                         input_shape=(224, 224, 3))

# Add a global average pooling layer and a dense layer for binary classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=20,  # You can adjust the number of epochs
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)

# Save the trained model
model.save('currency_detection_model.h5')
# Extract features from the base MobileNetV2 model for KNN and SVM
base_model_output = base_model.predict(validation_generator)

# Flatten the feature vectors
num_samples = base_model_output.shape[0]
base_model_output = base_model_output.reshape(num_samples, -1)

# Load labels for validation set
validation_labels = np.array(validation_generator.classes)

# Train a KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(base_model_output, validation_labels)

# Train an SVM classifier
svm = SVC(kernel='linear', C=1.0)
svm.fit(base_model_output, validation_labels)

# Evaluate the KNN and SVM models on the validation set
knn_predictions = knn.predict(base_model_output)
svm_predictions = svm.predict(base_model_output)

knn_accuracy = accuracy_score(validation_labels, knn_predictions)
svm_accuracy = accuracy_score(validation_labels, svm_predictions)

print(f'KNN Accuracy: {knn_accuracy:.2f}')

# Find the maximum accuracy from all epochs
max_accuracy = max(history.history['val_accuracy'])
print(f'Maximum Validation Accuracy: {max_accuracy:.2f}')

with open('accuracies.txt', 'w') as file:
    file.write(f'KNN Accuracy: {knn_accuracy:.2f}\n')
    file.write(f'CNN Accuracy: {max_accuracy:.2f}\n')
