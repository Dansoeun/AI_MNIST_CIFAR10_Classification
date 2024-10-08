import numpy as np
import tensorflow as tf
from keras import layers, models
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np

# MNIST Data
(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = keras.datasets.mnist.load_data()

# CIFAR-10 Data
(cifar_x_train, cifar_y_train), (cifar_x_test, cifar_y_test) = keras.datasets.cifar10.load_data()

# Reshape MNIST and normalize both datasets
x_train_mnist = x_train_mnist.astype(np.float32) / 255.0
x_test_mnist = x_test_mnist.astype(np.float32) / 255.0

# Reshape MNIST to (32, 32, 1) by padding to match CIFAR-10 size
x_train_mnist = np.pad(x_train_mnist, ((0, 0), (2, 2), (2, 2)), 'constant')
x_test_mnist = np.pad(x_test_mnist, ((0, 0), (2, 2), (2, 2)), 'constant')

# Add channel dimension to match CIFAR-10 format
x_train_mnist = x_train_mnist.reshape(-1, 32, 32, 1)
x_test_mnist = x_test_mnist.reshape(-1, 32, 32, 1)

# Convert CIFAR-10 to grayscale
def convert_to_grayscale(data):
    grayscale_data = []
    for img in data:
        grayscale_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        grayscale_data.append(grayscale_img)
    return np.array(grayscale_data)

cifar_x_train_gray = convert_to_grayscale(cifar_x_train)
cifar_x_test_gray = convert_to_grayscale(cifar_x_test)

# Normalize CIFAR-10 grayscale images
cifar_x_train_gray = cifar_x_train_gray.astype(np.float32) / 255.0
cifar_x_test_gray = cifar_x_test_gray.astype(np.float32) / 255.0

# Add channel dimension to CIFAR-10 grayscale
cifar_x_train_gray = cifar_x_train_gray.reshape(-1, 32, 32, 1)
cifar_x_test_gray = cifar_x_test_gray.reshape(-1, 32, 32, 1)

# Label MNIST as class 0 and CIFAR-10 as class 1
y_train_mnist_bin = np.zeros(len(y_train_mnist))
y_test_mnist_bin = np.zeros(len(y_test_mnist))

y_train_cifar_bin = np.ones(len(cifar_y_train))
y_test_cifar_bin = np.ones(len(cifar_y_test))

# Combine the MNIST and CIFAR-10 datasets
x_train_combined = np.concatenate((x_train_mnist, cifar_x_train_gray), axis=0)
y_train_combined = np.concatenate((y_train_mnist_bin, y_train_cifar_bin), axis=0)

x_test_combined = np.concatenate((x_test_mnist, cifar_x_test_gray), axis=0)
y_test_combined = np.concatenate((y_test_mnist_bin, y_test_cifar_bin), axis=0)

# Shuffle the combined datasets
x_train_combined, y_train_combined = shuffle(x_train_combined, y_train_combined, random_state=42)
x_test_combined, y_test_combined = shuffle(x_test_combined, y_test_combined, random_state=42)

# Model Creation
model = models.Sequential()

# First Convolutional Block
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))
model.add(layers.MaxPooling2D((2, 2)))

# Second Convolutional Block
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Third Convolutional Block
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten and Dense Layers
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification output

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train_combined, y_train_combined, epochs=10, batch_size=64,
                    validation_data=(x_test_combined, y_test_combined))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test_combined, y_test_combined)
print(f'Test accuracy: {test_acc}')

# Make predictions on the test set
predictions = model.predict(x_test_combined)

# Convert probabilities to class labels (0 or 1)
# For binary classification, you can use a threshold of 0.5
pred_classes = (predictions > 0.5).astype(int).flatten()  # Flatten to 1D array

# Since we labeled MNIST as 0 and CIFAR-10 as 1
true_classes = y_test_combined  # Already in binary form (0 for MNIST, 1 for CIFAR-10)

# Plotting the Actual vs. Predicted results
class_names = ['MNIST (0)', 'CIFAR-10 (1)']  # Adjust class names for binary classification

fig, axes = plt.subplots(5, 5, figsize=(15, 15))
axes = axes.ravel()

for i in np.arange(0, 25):  # Display 25 images (5x5 grid)
    axes[i].imshow(x_test_combined[i].reshape(32, 32), cmap='gray')  # Ensure the image is displayed correctly for grayscale
    # Use int to ensure proper indexing
    axes[i].set_title("True: %s \nPredict: %s" % (class_names[int(true_classes[i])], class_names[int(pred_classes[i])]))
    axes[i].axis('off')

plt.subplots_adjust(wspace=1)
plt.show()