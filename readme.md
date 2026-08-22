Below is a polished **`README.md`** version with rich Markdown formatting, section hierarchy, code blocks, tables, notes, and a clear training workflow. I also completed the truncated **“Load the Best Model”** section.

# 🥭 Mango Leaf Classification using MobileNetV2

A deep learning pipeline for **mango leaf disease/stress classification** using the **MangoLeafBD Dataset** and **MobileNetV2 transfer learning**.

The workflow includes:

* 📂 Dataset preparation
* 🔀 Train/validation/test splitting
* ⚡ Efficient TensorFlow data pipelines
* 🖼️ Training image visualization
* 🧠 MobileNetV2 transfer learning
* 🔄 Data augmentation
* 🏋️ Initial model training
* 🎯 Fine-tuning
* 💾 Best-model checkpointing

---

## 📋 Dataset

This project uses the **MangoLeafBD Dataset**, containing approximately **4,000 mango leaf images** distributed across multiple classes.

The expected dataset structure is:

```text
MangoLeafBD Dataset/
├── Class_1/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
├── Class_2/
│   ├── image001.jpg
│   └── ...
├── Class_3/
│   └── ...
└── ...
```

Each subdirectory represents one classification class.

---

# 🚀 1. Set the Dataset Path

```python
from pathlib import Path
import pandas as pd

DATASET_PATH = Path("/content/dataset/MangoLeafBD Dataset")

print("Dataset exists:", DATASET_PATH.exists())

classes = sorted([
    p.name for p in DATASET_PATH.iterdir()
    if p.is_dir()
])

print("Classes:")

for i, class_name in enumerate(classes):
    print(f"{i}: {class_name}")

print("\nNumber of classes:", len(classes))
```

### Expected Output

```text
Dataset exists: True

Classes:
0: Class_1
1: Class_2
2: Class_3
...

Number of classes: N
```

> **Note:** Make sure the dataset path matches the location where the dataset has been uploaded or mounted in Google Colab.

---

# 🔀 2. Create Training, Validation, and Test Datasets

The dataset is divided into:

| Dataset       | Percentage |
| ------------- | ---------: |
| 🏋️ Training  |    **70%** |
| 🔎 Validation |    **15%** |
| 🧪 Testing    |    **15%** |
| **Total**     |   **100%** |

We first reserve **30%** of the dataset and then split that portion equally between validation and testing.

```python
import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
```

### Create the Training Dataset

```python
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.30,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)
```

### Create the Temporary Dataset

```python
temp_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.30,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

class_names = train_ds.class_names

print("Classes:", class_names)
```

The temporary dataset contains the remaining **30%** of the images.

We now divide it equally:

```python
temp_batches = tf.data.experimental.cardinality(temp_ds)

test_size = int(temp_batches * 0.5)

test_ds = temp_ds.take(test_size)
val_ds = temp_ds.skip(test_size)

print(
    "Train batches:",
    tf.data.experimental.cardinality(train_ds).numpy()
)

print(
    "Validation batches:",
    tf.data.experimental.cardinality(val_ds).numpy()
)

print(
    "Test batches:",
    tf.data.experimental.cardinality(test_ds).numpy()
)
```

> ⚠️ **Important:** Because the split is performed at the batch level, the exact number of images in validation and testing can differ slightly from exactly 15%.

---

# ⚡ 3. Improve Data Pipeline Performance

TensorFlow's `AUTOTUNE` is used to optimize data loading and preprocessing.

```python
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)
```

This allows data loading and model computation to overlap, improving training efficiency.

---

# 🖼️ 4. Visualize Training Images

Before training the model, it is useful to inspect sample images and verify their corresponding labels.

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 10))

for images, labels in train_ds.take(1):

    for i in range(12):

        plt.subplot(3, 4, i + 1)

        plt.imshow(
            images[i].numpy().astype("uint8")
        )

        plt.title(class_names[labels[i]])
        plt.axis("off")

plt.tight_layout()
plt.show()
```

This visualization helps verify:

* Image loading
* Image dimensions
* Class labels
* Dataset quality
* Possible image anomalies

---

# 🧠 5. Build the Model Using MobileNetV2

For a dataset of approximately **4,000 images**, transfer learning provides an effective way to leverage features learned from a large image dataset.

We use **MobileNetV2** pretrained on **ImageNet**.

### Import Required Components

```python
from tensorflow.keras.applications import MobileNetV2

from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    BatchNormalization
)

from tensorflow.keras.models import Model
```

### Load the Pretrained Base Model

```python
NUM_CLASSES = len(class_names)

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers initially
base_model.trainable = False
```

Initially, the pretrained MobileNetV2 layers are frozen so that only the newly added classification layers are trained.

---

## 🔄 Data Augmentation

Data augmentation helps improve model generalization by exposing the model to different variations of training images.

```python
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
])
```

The augmentation operations include:

| Augmentation    | Purpose                               |
| --------------- | ------------------------------------- |
| Random Flip     | Simulates different leaf orientations |
| Random Rotation | Handles changes in leaf angle         |
| Random Zoom     | Improves scale robustness             |
| Random Contrast | Handles lighting variation            |

---

## 🏗️ Create the Complete Model

```python
inputs = tf.keras.Input(
    shape=(224, 224, 3)
)

x = data_augmentation(inputs)

# MobileNetV2 preprocessing
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(
    x,
    training=False
)

x = GlobalAveragePooling2D()(x)

x = BatchNormalization()(x)

x = Dropout(0.3)(x)

outputs = Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = Model(
    inputs,
    outputs
)

model.summary()
```

### Model Architecture

```text
Input Image
   │
   ▼
224 × 224 × 3
   │
   ▼
Data Augmentation
   │
   ▼
MobileNetV2 Preprocessing
   │
   ▼
MobileNetV2 Backbone
   │
   ▼
Global Average Pooling
   │
   ▼
Batch Normalization
   │
   ▼
Dropout (0.3)
   │
   ▼
Dense Layer
   │
   ▼
Softmax Classification
   │
   ▼
Mango Leaf Class
```

---

# ⚙️ 6. Compile the Model

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

### Configuration

| Parameter             | Value                           |
| --------------------- | ------------------------------- |
| Optimizer             | Adam                            |
| Initial Learning Rate | `0.001`                         |
| Loss                  | Sparse Categorical Crossentropy |
| Metric                | Accuracy                        |

---

# 🛡️ 7. Add Training Callbacks

Callbacks help prevent overfitting and automatically preserve the best-performing model.

```python
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

callbacks = [

    ModelCheckpoint(
        "/content/best_mango_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
]
```

### Callback Functions

| Callback            | Purpose                                                    |
| ------------------- | ---------------------------------------------------------- |
| `ModelCheckpoint`   | Saves the best model                                       |
| `EarlyStopping`     | Stops training when validation performance stops improving |
| `ReduceLROnPlateau` | Reduces the learning rate when validation loss stagnates   |

The best model is saved as:

```text
/content/best_mango_model.keras
```

---

# 🏋️ 8. Train the Model

The first training stage trains only the newly added classification layers while the MobileNetV2 backbone remains frozen.

```python
EPOCHS = 20

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)
```

### Training Strategy

```text
MangoLeafBD Dataset
        │
        ▼
   70% Training
        │
        ▼
Data Augmentation
        │
        ▼
MobileNetV2 Frozen
        │
        ▼
Classification Head Training
        │
        ▼
Validation
        │
        ▼
Best Model Checkpoint
```

The best model will automatically be saved at:

```text
/content/best_mango_model.keras
```

---

# 🎯 9. Fine-Tune the Model

After the initial training phase, part of the pretrained MobileNetV2 network can be unfrozen.

This allows the model to adapt some of its pretrained visual features specifically to mango leaf images.

```python
base_model.trainable = True

# Freeze most layers
for layer in base_model.layers[:-30]:
    layer.trainable = False
```

Only the final **30 layers** of MobileNetV2 are allowed to update.

---

## 🔧 Compile with a Lower Learning Rate

Fine-tuning requires a much smaller learning rate to avoid destroying useful pretrained features.

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

---

## 🏃 Train the Fine-Tuned Model

```python
fine_tune_history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=callbacks
)
```

### Fine-Tuning Strategy

```text
Pretrained MobileNetV2
        │
        ▼
Freeze Most Layers
        │
        ▼
Unfreeze Last 30 Layers
        │
        ▼
Lower Learning Rate
        │
        ▼
Fine-Tuning
        │
        ▼
Validation
        │
        ▼
Best Model
```

---

# 💾 10. Load the Best Model

After training, load the checkpoint that achieved the highest validation accuracy.

```python
best_model = tf.keras.models.load_model(
    "/content/best_mango_model.keras"
)

print("Best model loaded successfully.")
```

Verify the model:

```python
best_model.summary()
```

---

# 🧪 11. Evaluate the Model on the Test Dataset

The test dataset should be used only for the final unbiased evaluation.

```python
test_loss, test_accuracy = best_model.evaluate(
    test_ds,
    verbose=1
)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
```

The reported test accuracy represents the model's performance on previously unseen images.

---

# 📊 12. Plot Training History

Plot training and validation accuracy:

```python
plt.figure(figsize=(10, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)

plt.show()
```

Plot training and validation loss:

```python
plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)

plt.show()
```

---

# 🔍 13. Generate Predictions

Predictions can be generated for the test dataset using:

```python
import numpy as np

y_true = []
y_pred = []

for images, labels in test_ds:

    predictions = best_model.predict(
        images,
        verbose=0
    )

    predicted_labels = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels)

y_true = np.array(y_true)
y_pred = np.array(y_pred)
```

---

# 📋 14. Classification Report

A detailed classification report can be generated using `scikit-learn`.

```python
from sklearn.metrics import classification_report

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)
```

The report provides:

* Precision
* Recall
* F1-score
* Support

for every class.

---

# 🧩 15. Confusion Matrix

```python
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(figsize=(10, 8))

plt.imshow(cm)

plt.colorbar()

plt.xticks(
    range(len(class_names)),
    class_names,
    rotation=90
)

plt.yticks(
    range(len(class_names)),
    class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

plt.tight_layout()
plt.show()
```

The confusion matrix helps identify which mango leaf classes are being confused by the model.

---

# 🔮 16. Predict a Single Image

A single mango leaf image can be classified using:

```python
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
import numpy as np

IMAGE_PATH = "/content/sample_mango_leaf.jpg"

image = load_img(
    IMAGE_PATH,
    target_size=IMG_SIZE
)

image_array = img_to_array(image)

image_array = np.expand_dims(
    image_array,
    axis=0
)

prediction = best_model.predict(
    image_array
)

predicted_class = class_names[
    np.argmax(prediction[0])
]

confidence = np.max(
    prediction[0]
)

print("Predicted Class:", predicted_class)
print(
    f"Confidence: {confidence * 100:.2f}%"
)
```

### Example Output

```text
Predicted Class: Class_3
Confidence: 94.72%
```

---

# 📦 17. Save the Final Model

The trained model can be exported for deployment.

```python
best_model.save(
    "/content/mango_leaf_classifier.keras"
)
```

The resulting file:

```text
mango_leaf_classifier.keras
```

can later be used for inference in a web application, mobile application, API, or agricultural decision-support system.

---

# 🗂️ Project Structure

A recommended project structure is:

```text
mango-leaf-classification/
│
├── README.md
│
├── dataset/
│   └── MangoLeafBD Dataset/
│       ├── Class_1/
│       ├── Class_2/
│       └── ...
│
├── notebooks/
│   └── mango_leaf_classification.ipynb
│
├── models/
│   └── best_mango_model.keras
│
├── results/
│   ├── training_history.png
│   ├── confusion_matrix.png
│   └── classification_report.txt
│
└── inference/
    └── predict.py
```

---

# 🧠 Overall Workflow

```text
                 MangoLeafBD Dataset
                          │
                          ▼
                Dataset Verification
                          │
                          ▼
                 70 / 15 / 15 Split
                          │
             ┌────────────┼────────────┐
             ▼            ▼            ▼
          Training    Validation      Test
             │
             ▼
       Data Augmentation
             │
             ▼
        MobileNetV2
       ImageNet Weights
             │
             ▼
       Frozen Backbone
             │
             ▼
     Classification Head
             │
             ▼
       Initial Training
             │
             ▼
        Fine-Tuning
             │
             ▼
       Best Checkpoint
             │
             ▼
        Test Evaluation
             │
             ▼
       Final Classifier
```

---

# ⚙️ Key Hyperparameters

| Parameter                 |          Setting |
| ------------------------- | ---------------: |
| Image Size                |      `224 × 224` |
| Batch Size                |             `32` |
| Random Seed               |             `42` |
| Training Split            |            `70%` |
| Validation Split          |            `15%` |
| Test Split                |            `15%` |
| Backbone                  |      MobileNetV2 |
| Pretrained Weights        |         ImageNet |
| Initial Learning Rate     |          `0.001` |
| Fine-Tuning Learning Rate |        `0.00001` |
| Initial Epochs            |             `20` |
| Fine-Tuning Epochs        |             `10` |
| Dropout                   |            `0.3` |
| Fine-Tuned Layers         | Last `30` layers |

---

# 📌 Notes

### Dataset Balance

Before training, it is recommended to inspect the number of images in each class. Significant class imbalance can negatively affect model performance.

### Data Leakage

Images originating from the same physical leaf, plant, or capture session should ideally not appear across training, validation, and test sets.

### Test Set Isolation

The test set should remain untouched during model development and hyperparameter tuning. It should be used only for final evaluation.

### Fine-Tuning

Fine-tuning should use a significantly lower learning rate than the initial transfer-learning stage.

---

# 🚀 Future Improvements

Potential extensions include:

* 🔬 Class-specific disease severity estimation
* 🧠 Explainable AI using Grad-CAM
* 📱 Mobile deployment using TensorFlow Lite
* 🌐 REST API deployment
* 📷 Real-time camera-based detection
* 🌦️ Integration with weather and environmental data
* 🌱 Crop-stage-aware disease prediction
* 🤖 LLM-based farmer advisory
* 📈 Continuous learning using newly collected field images

---

# 📄 License

Add the appropriate dataset and project license here.

---

# 👨‍💻 Project

**Mango Leaf Classification using Deep Transfer Learning**

Built with:

* Python
* TensorFlow / Keras
* MobileNetV2
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
