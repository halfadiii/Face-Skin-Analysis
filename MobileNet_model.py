import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

# Ensure you have TensorFlow 2.x installed
print("TensorFlow version:", tf.__version__)

def create_model(num_classes):
    # Load the MobileNetV2 model, pre-trained on ImageNet, without the top classification layer
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the convolutional base to prevent weights from being updated during training
    base_model.trainable = False

    # Create the custom top layers for our classification task
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # Combine the base model and the custom layers into a new model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    # Define the number of classes
    num_classes = 7  # Update this according to the number of folders/classes in your dataset

    # Path to your training data
    train_data_dir = r'C:\Adi\GitHub\LOREAL\images\archive\DATA\train'

    # Set up data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

    # Create the model
    model = create_model(num_classes)

    # Train the model
    steps_per_epoch = train_generator.samples // train_generator.batch_size
    model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=100)

    # Save the model
    model.save(r'Models\my_mobilenet_model.h5')
    print("Model saved as my_mobilenet_model.h5")

if __name__ == "__main__":
    main()
