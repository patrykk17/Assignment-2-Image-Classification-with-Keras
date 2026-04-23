from __future__ import print_function

import os
import keras
import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Rescaling, BatchNormalization
from keras.optimizers import RMSprop, Adam
import matplotlib.pyplot as plt
import numpy as np
import time

batch_size = 12
num_classes = 3
epochs = 8
img_width = 128
img_height = 128
img_channels = 3
fit = True

train_dir = 'C:\\Users\\patry\\Desktop\\Assignment2\\chest_xray\\train'
test_dir = 'C:\\Users\\patry\\Desktop\\Assignment2\\chest_xray\\test'

train_ds, val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    seed=123,
    validation_split=0.2,
    subset='both',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    labels='inferred',
    shuffle=True)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    seed=None,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    labels='inferred',
    shuffle=True)

class_names = train_ds.class_names
print('Class Names:', class_names)
num_classes = len(class_names)

# ── DATASET ANALYSIS - CLASS DISTRIBUTION (Q2) ────────────────────────────────
print("\nClass Distribution in Training Set:")
train_counts = []
for cls in class_names:
    count = len(os.listdir(os.path.join(train_dir, cls)))
    train_counts.append(count)
    print(f"  {cls}: {count} images")
print(f"  Total: {sum(train_counts)} images")

print("\nClass Distribution in Test Set:")
for cls in class_names:
    count = len(os.listdir(os.path.join(test_dir, cls)))
    print(f"  {cls}: {count} images")

plt.figure(figsize=(8, 5))
plt.bar(class_names, train_counts, color=['steelblue', 'orange', 'green'])
plt.title('Training Set Class Distribution')
plt.ylabel('Number of Images')
plt.xlabel('Class')
plt.tight_layout()
plt.show()

# ── CLASS WEIGHTS TO ADDRESS IMBALANCE (Q2) ───────────────────────────────────
total = sum(train_counts)
class_weight = {i: total / (num_classes * train_counts[i]) for i in range(num_classes)}
print("\nClass Weights:", class_weight)

# ── SAMPLE IMAGES ─────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(2):
    for i in range(6):
        ax = plt.subplot(2, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i].numpy()])
        plt.axis("off")
plt.show()

# ── BUILD MODEL - replace Flatten with GlobalAveragePooling2D to reduce overfitting (Q3) ──
model = tf.keras.models.Sequential([
    Rescaling(1.0/255),
    Conv2D(16, (3,3), activation='relu', input_shape=(img_height, img_width, img_channels)),
    MaxPooling2D(2,2),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    tf.keras.layers.GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

model.summary()

save_callback = tf.keras.callbacks.ModelCheckpoint("pneumonia.keras", save_freq='epoch', save_best_only=True)

start = time.time()
if fit:
    history = model.fit(
        train_ds,
        batch_size=batch_size,
        validation_data=val_ds,
        callbacks=[save_callback],
        class_weight=class_weight,
        epochs=epochs)
else:
    model = tf.keras.models.load_model("pneumonia.keras")
end = time.time()
print(f'\nTraining time: {end - start:.2f} seconds')

score = model.evaluate(test_ds, batch_size=batch_size)
print('Test accuracy:', score[1])

# ── ACCURACY + LOSS PLOTS (Q3) ────────────────────────────────────────────────
if fit:
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.show()

# ── SAMPLE PREDICTIONS ────────────────────────────────────────────────────────
test_batch = test_ds.take(1)
plt.figure(figsize=(10, 10))
for images, labels in test_batch:
    for i in range(6):
        ax = plt.subplot(2, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        prediction = model.predict(tf.expand_dims(images[i].numpy(), 0), verbose=0)
        plt.title('Actual:' + class_names[labels[i].numpy()] + '\nPredicted:{} {:.2f}%'.format(class_names[np.argmax(prediction)], 100 * np.max(prediction)))
        plt.axis("off")
plt.show()