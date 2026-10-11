import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42


def main():
    parser = argparse.ArgumentParser(description="Train a CNN on a folder-based image dataset.")
    parser.add_argument("--data", default="dataset/yawn_eye", help="Folder containing one subfolder per class")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    data_dir = Path(args.data)
    if not data_dir.exists():
        raise SystemExit(
            f"Dataset folder not found: {data_dir}\n"
            "Create it and place one class folder per label inside it. See README.md."
        )

    # image_dataset_from_directory reads immediate subfolders as class labels.
    all_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=None,
        shuffle=False,
    )
    class_names = all_ds.class_names
    if len(class_names) < 2:
        raise SystemExit("Need at least two class folders containing images.")

    paths = np.array(all_ds.file_paths)
    # Reload labels in the same deterministic order using the dataset pipeline.
    y = np.concatenate([label.numpy().reshape(-1) for _, label in all_ds.batch(512)], axis=0)
    indices = np.arange(len(paths))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.30, random_state=SEED, stratify=y
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=SEED, stratify=y[temp_idx]
    )

    # Build datasets from file paths and labels, so the split is reproducible.
    def load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMAGE_SIZE)
        return image, label

    def make_ds(idx, training=False):
        ds = tf.data.Dataset.from_tensor_slices((paths[idx], y[idx]))
        if training:
            ds = ds.shuffle(min(len(idx), 2000), seed=SEED)
        ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = make_ds(train_idx, True)
    val_ds = make_ds(val_idx)
    test_ds = make_ds(test_idx)

    num_classes = len(class_names)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3)),
        tf.keras.layers.Rescaling(1.0 / 255),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.30),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("models/yawn_eye_cnn.keras", monitor="val_loss", save_best_only=True),
    ]
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    loss, accuracy = model.evaluate(test_ds, verbose=0)
    print(f"\nTest loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")

    true_labels, predicted_labels = [], []
    for images, labels_batch in test_ds:
        probs = model.predict(images, verbose=0)
        true_labels.extend(labels_batch.numpy().tolist())
        predicted_labels.extend(np.argmax(probs, axis=1).tolist())

    report = classification_report(
        true_labels, predicted_labels, labels=list(range(num_classes)),
        target_names=class_names, zero_division=0
    )
    print("\nClassification report:\n", report)
    with open("outputs/classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    with open("models/class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    cm = confusion_matrix(true_labels, predicted_labels, labels=list(range(num_classes)))
    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(range(num_classes), class_names, rotation=45, ha="right")
    plt.yticks(range(num_classes), class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(history.history["accuracy"], label="Train accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/accuracy.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="Train loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/loss.png", dpi=150)
    plt.close()

    print("\nSaved model: models/yawn_eye_cnn.keras")
    print("Saved class names: models/class_names.json")
    print("Saved reports and plots in outputs/")


if __name__ == "__main__":
    main()
