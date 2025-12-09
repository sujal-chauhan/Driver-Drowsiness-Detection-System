import os
import json
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

# ----------------- Paths -----------------
BASE_DIR = r'C:\Users\Tushar\Documents\GitHub\drowsiness-detection'
TRAIN_DIR = os.path.join(BASE_DIR, 'data', 'train')
VAL_DIR   = os.path.join(BASE_DIR, 'data', 'validation')
TEST_DIR  = os.path.join(BASE_DIR, 'data', 'test')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# ----------------- Hyperparameters -----------------
IMG_SIZE = (24, 24)
BATCH_SIZE = 32
EPOCHS = 4

# ----------------- Data generators -----------------
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1.0/255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_gen = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

num_classes = len(train_gen.class_indices)
print("Class indices:", train_gen.class_indices)

# ----------------- Model -----------------
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Dropout(0.3),
    Flatten(),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),

    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ----------------- Callbacks -----------------
checkpoint_path = os.path.join(MODEL_DIR, 'cnnCat2.h5')  # used by drowsiness-detection.py
checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=6,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# ----------------- Training -----------------
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=[checkpoint, early_stop, reduce_lr]
)

# ----------------- Evaluation on train & validation -----------------
print("Evaluating on training data...")
train_loss, train_acc = model.evaluate(train_gen, verbose=0)
print(f"Train accuracy: {train_acc * 100:.2f}%")

print("Evaluating on validation data...")
val_loss, val_acc = model.evaluate(val_gen, verbose=0)
print(f"Validation accuracy: {val_acc * 100:.2f}%")

# ----------------- Save metrics -----------------
metrics = {
    "train_loss": float(train_loss),
    "train_accuracy": float(train_acc),
    "val_loss": float(val_loss),
    "val_accuracy": float(val_acc),
    "class_indices": train_gen.class_indices
}

with open(os.path.join(MODEL_DIR, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=4)

# ----------------- Plot training history -----------------
def plot_history(history, out_path):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Train Acc')
    plt.plot(epochs_range, val_acc, label='Val Acc')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Val Loss')
    plt.legend()
    plt.title('Loss')

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

plot_history(history, os.path.join(MODEL_DIR, 'training_history.png'))
