from __future__ import print_function

import os
import tensorflow as tf
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.metrics import classification_report

batch_size = 32
num_classes = 3
epochs = 15
img_width = 128
img_height = 128
fit = False

train_dir = 'C:\\Users\\patry\\Desktop\\Assignment2\\chest_xray\\train'
test_dir = 'C:\\Users\\patry\\Desktop\\Assignment2\\chest_xray\\test'

AUTOTUNE = tf.data.AUTOTUNE

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

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

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

# ── CLASS WEIGHTS (Q2) ────────────────────────────────────────────────────────
total = sum(train_counts)
class_weight = {i: total / (num_classes * train_counts[i]) for i in range(num_classes)}
print("\nClass Weights:", class_weight)

# ── SAMPLE IMAGES ─────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(6):
        ax = plt.subplot(2, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i].numpy()])
        plt.axis("off")
plt.show()

# ── TRANSFER LEARNING WITH EfficientNetB0 (Q6) ────────────────────────────────
base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet')
base_model.trainable = False

inputs = tf.keras.Input(shape=(img_height, img_width, 3))
x = tf.keras.layers.RandomFlip("horizontal")(inputs)
x = tf.keras.layers.RandomRotation(0.1)(x)
x = tf.keras.layers.RandomZoom(0.1)(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])

model.summary()

save_callback = tf.keras.callbacks.ModelCheckpoint("pneumonia.keras", save_freq='epoch', save_best_only=True)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True)

start = time.time()
if fit:
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        callbacks=[save_callback, early_stop],
        epochs=epochs)

    # ── FINE-TUNING (Q6) ──────────────────────────────────────────────────────
    print("\nFine-tuning top layers of EfficientNetB0...")
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.00001),
                  metrics=['accuracy'])

    start_fine = time.time()
    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=8)
    end_fine = time.time()
    print(f'Fine-tuning time: {end_fine - start_fine:.2f} seconds')

    # ── PLOTS ─────────────────────────────────────────────────────────────────
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.title('Model Accuracy - EfficientNetB0')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.title('Model Loss - EfficientNetB0')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history_fine.history['accuracy'], label='train')
    plt.plot(history_fine.history['val_accuracy'], label='val')
    plt.title('Fine-tuning Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history_fine.history['loss'], label='train')
    plt.plot(history_fine.history['val_loss'], label='val')
    plt.title('Fine-tuning Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.show()

else:
    model = tf.keras.models.load_model("pneumonia.keras")

end = time.time()
print(f'\nTotal time: {end - start:.2f} seconds')

score = model.evaluate(test_ds, batch_size=batch_size)
print('Test accuracy:', score[1])

# ── PER-CLASS METRICS (Q7, Q8) ────────────────────────────────────────────────
y_true = []
y_pred = []
for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# ── SAMPLE PREDICTIONS ────────────────────────────────────────────────────────
test_batch = test_ds.take(1)
plt.figure(figsize=(10, 10))
for images, labels in test_batch:
    for i in range(6):
        ax = plt.subplot(2, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        prediction = model.predict(tf.expand_dims(images[i].numpy(), 0), verbose=0)
        plt.title('Actual:' + class_names[labels[i].numpy()] +
                  '\nPredicted:{} {:.2f}%'.format(class_names[np.argmax(prediction)], 100 * np.max(prediction)))
        plt.axis("off")
plt.show()