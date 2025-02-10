import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import tensorflow as tf
from tensorflow.keras.applications import ResNet50# type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D# type: ignore
from tensorflow.keras.models import Model# type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator# type: ignore

# Ensure TensorFlow is installed
print("TensorFlow version:", tf.__version__)

def create_model(num_classes):
    # Load ResNet50 pre-trained on ImageNet, without the top layer
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze the layers of the base model to prevent them from being updated during training
    base_model.trainable = False

    # Add custom layers on top of ResNet50 for our specific task
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Reduce feature map dimensions
    x = Dense(1024, activation='relu')(x)  # A dense layer for learning new features
    predictions = Dense(num_classes, activation='softmax')(x)  # Final layer with softmax activation

    # Complete the model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    num_classes = 7  # Set this to the number of skin condition classes you have

    # Data generators for training and validation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    validation_datagen = ImageDataGenerator(rescale=1./255)

    # Replace 'path_to_train' and 'path_to_validation' with your dataset directories
    train_generator = train_datagen.flow_from_directory(
        r'C:\Adi\GitHub\LOREAL\images\archive\DATA\train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

    validation_generator = validation_datagen.flow_from_directory(
        r'C:\Adi\GitHub\LOREAL\images\archive\DATA\testing',
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
    model.save(r'Models\skin_condition_model_resnet50.h5')
    print("Model saved as skin_condition_model_resnet50.h5")

if __name__ == '__main__':
    main()
