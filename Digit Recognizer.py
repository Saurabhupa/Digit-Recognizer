import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

# Load the training and test datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Extract labels from training data
labels = train_data['label']

# Extract and normalize pixel values
features = train_data.drop('label', axis=1) / 255.0

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

# Convert labels to one-hot encoded format
y_train = to_categorical(y_train, num_classes=10)
y_val = to_categorical(y_val, num_classes=10)

# Create a convolutional neural network model
model = Sequential()
# Add layers to the model (you can experiment with different architectures)
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train.values.reshape(-1, 28, 28, 1), y_train, epochs=10, batch_size=128, validation_data=(X_val.values.reshape(-1, 28, 28, 1), y_val))

# Evaluate the model on the validation set
accuracy = model.evaluate(X_val.values.reshape(-1, 28, 28, 1), y_val)[1]
print("Validation Accuracy:", accuracy)

# Make predictions on the test data
X_test = test_data.values / 255.0
X_test = X_test.reshape(-1, 28, 28, 1)
predictions = model.predict(X_test)

# Convert predictions to labels
predicted_labels = np.argmax(predictions, axis=1)

# Create a submission DataFrame
submission = pd.DataFrame({'ImageId': range(1, len(predicted_labels) + 1), 'Label': predicted_labels})

# Save the submission to a CSV file
submission.to_csv('submission.csv', index=False)
