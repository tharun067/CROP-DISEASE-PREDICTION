import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the image size and batch size
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 15

# Create an instance of ImageDataGenerator for validation
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Create validation generator
validation_generator = datagen.flow_from_directory(
    'PlantVillage2/',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Load the trained model
model = tf.keras.models.load_model('models/crop_disease_model_resnet50.h5')

# Evaluate the model
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f'Validation Loss: {val_loss}')
print(f'Validation Accuracy: {val_accuracy}')
