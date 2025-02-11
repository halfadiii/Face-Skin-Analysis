import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0# type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D# type: ignore
from tensorflow.keras.models import Model# type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator# type: ignore

# Ensure TensorFlow is installed
print("TensorFlow version:", tf.__version__)

def create_model(num_classes):
    # Load EfficientNetB0 pre-trained on ImageNet, exclude the top layer
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze the layers of the base model to prevent them from being updated during training
    base_model.trainable = False

    # Add custom layers on top of the base model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)  # Adjust the number of output classes

    # Construct the full model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    num_classes = 7  # Update this to match the number of skin condition classes in your dataset

    # Data generators with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        r'C:\Adi\GitHub\LOREAL\images\archive\DATA\train',  # Specify the path to your training data
        target_size=(224, 224),  # EfficientNetB0 uses 224x224 inputs
        batch_size=32,
        class_mode='categorical')

    validation_generator = validation_datagen.flow_from_directory(
        r'C:\Adi\GitHub\LOREAL\images\archive\DATA\testing',  # Specify the path to your validation data
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

    # Create and train the model
    model = create_model(num_classes)
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size)

    # Save the trained model
    model.save(r'Models\skin_condition_model_efficientnetb0.h5')
    print("Model saved as skin_condition_model_efficientnetb0.h5")

if __name__ == '__main__':
    main()
