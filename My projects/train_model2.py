import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import json

# Define the image size and batch size
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 15

# Print TensorFlow version for compatibility check
print(f"TensorFlow Version: {tf.__version__}")

# Create an instance of ImageDataGenerator for training and validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Create training and validation generators
train_generator = train_datagen.flow_from_directory(
    'PlantVillage2/',  
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'PlantVillage2/',  
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Calculate class weights to handle imbalance
class_labels = list(train_generator.class_indices.keys())
num_classes = len(class_labels)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))

# Define input tensor
input_tensor = Input(shape=IMAGE_SIZE + (3,))

# Load ResNet50 model
base_model = ResNet50(input_tensor=input_tensor, include_top=False, weights='imagenet')

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output_tensor = Dense(num_classes, activation='softmax')(x)

# Construct the model
model = Model(inputs=base_model.input, outputs=output_tensor)

# Print model summary
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with class weights
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    class_weight=class_weights
)

# Save the trained model
model.save('models/crop_disease_model_resnet50.h5')

# Save the class labels
with open('models/class_labels.json', 'w') as f:
    json.dump(train_generator.class_indices, f)
