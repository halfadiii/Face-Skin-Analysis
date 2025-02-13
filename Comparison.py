import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator# type: ignore
from tensorflow.keras.models import load_model# type: ignore
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

model_details = {
    'MobileNetV2': {'path': r'C:\Adi\GitHub\LOREAL\Models\my_mobilenet_model.h5', 'size': (224, 224)},
    'EfficientNetB0': {'path': r'C:\Adi\GitHub\LOREAL\Models\skin_condition_model_efficientnetb0.h5', 'size': (224, 224)},
    'InceptionV3': {'path': r'C:\Adi\GitHub\LOREAL\Models\skin_condition_model_inceptionv3.h5', 'size': (299, 299)},
    'ResNet50': {'path': r'C:\Adi\GitHub\LOREAL\Models\skin_condition_model_resnet50.h5', 'size': (224, 224)}
}

def evaluate_model(model_path, target_size):
    print(f"Evaluating model at {model_path} with target size {target_size}...")
    model = load_model(model_path)
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        r'C:\Adi\GitHub\LOREAL\images\archive\DATA\testing',
        target_size=target_size,
        batch_size=32,
        class_mode='categorical',
        shuffle=False)

    loss, accuracy = model.evaluate(test_generator)
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    print(f"Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print(f"Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels, zero_division=0))
    
    return accuracy, class_labels

# Collect data
accuracies = []
model_names = []

for model_name, details in model_details.items():
    accuracy, labels = evaluate_model(details['path'], details['size'])
    accuracies.append(accuracy)
    model_names.append(model_name)

# Plotting
plt.figure(figsize=(10, 5))
plt.bar(model_names, accuracies, color='blue')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Comparison of Model Accuracies')
plt.ylim([0, 1])  # Set the limit of y-axis to show accuracies from 0 to 100%
plt.show()
