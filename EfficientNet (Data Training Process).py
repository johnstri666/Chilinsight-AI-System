import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB4

import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import seaborn as sns
import random

from sklearn.metrics import confusion_matrix, classification_report


# ==========================
# 0. REPRODUCIBILITY SEED
# ==========================
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)


# ==========================
# 1. DATASET PATHS
# ==========================
train_dir = "CHILIDISEASEDATASET/train"
val_dir   = "CHILIDISEASEDATASET/val"
test_dir  = "CHILIDISEASEDATASET/test"


# ==========================
# 2. PARAMETERS
# ==========================
IMG_SIZE   = (380, 380)   # EfficientNetB4 optimal input size
BATCH_SIZE = 16           # Smaller batch size for B4 (memory constraints)
EPOCHS     = 50


# ==========================
# 3. DATA AUGMENTATION
# EfficientNetB4 includes internal rescaling
# ==========================
train_datagen = ImageDataGenerator(
    rotation_range=25,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    fill_mode='nearest'
)

val_datagen  = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=1,
    shuffle=False,
    class_mode="categorical"
)

class_names = list(test_data.class_indices.keys())
num_classes = len(class_names)


# ==========================
# DATASET INFORMATION PER CLASS
# ==========================
def count_images_per_class(data_dir, class_names):
    """Count images in each class directory"""
    result = {}
    for cls in class_names:
        folder = os.path.join(data_dir, cls)
        if os.path.exists(folder):
            count = len([
                f for f in os.listdir(folder)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
            ])
        else:
            count = 0
        result[cls] = count
    return result

train_per_class = count_images_per_class(train_dir, class_names)
val_per_class   = count_images_per_class(val_dir,   class_names)
test_per_class  = count_images_per_class(test_dir,  class_names)

col_width = max(len(c) for c in class_names) + 2

print("\n" + "=" * 65)
print("              DATASET INFORMATION")
print("=" * 65)
print(f"  Number of Classes : {num_classes}")
print(f"  Class Names       : {class_names}")
print("-" * 65)
print(f"  {'Class':<{col_width}} {'Train':>8}  {'Val':>8}  {'Test':>8}  {'Total':>8}")
print("-" * 65)

total_train = total_val = total_test = 0
for cls in class_names:
    t  = train_per_class[cls]
    v  = val_per_class[cls]
    ts = test_per_class[cls]
    total_train += t
    total_val   += v
    total_test  += ts
    print(f"  {cls:<{col_width}} {t:>8}  {v:>8}  {ts:>8}  {t+v+ts:>8}")

print("-" * 65)
grand_total = total_train + total_val + total_test
print(f"  {'TOTAL':<{col_width}} {total_train:>8}  {total_val:>8}  {total_test:>8}  {grand_total:>8}")
print("=" * 65)
print(f"\n  Split Proportions:")
print(f"    Train : {total_train/grand_total*100:.1f}%  ({total_train} images)")
print(f"    Val   : {total_val/grand_total*100:.1f}%  ({total_val} images)")
print(f"    Test  : {total_test/grand_total*100:.1f}%  ({total_test} images)")
print("=" * 65 + "\n")


# ==========================
# 4. MODEL — EfficientNetB4 + Transfer Learning
# ==========================

# --- PHASE 1: FEATURE EXTRACTION ---
base_model = EfficientNetB4(
    include_top=False,
    weights="imagenet",
    input_shape=(380, 380, 3)
)
base_model.trainable = False    # Freeze all base layers

print(f"Total EfficientNetB4 layers  : {len(base_model.layers)}")
print(f"Frozen layers                : {len(base_model.layers)} layers\n")

# Build complete model
inp = Input(shape=(380, 380, 3))

x = base_model(inp, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
out = layers.Dense(num_classes, activation="softmax")(x)

model = tf.keras.Model(inputs=inp, outputs=out)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()
print(f"\nTotal parameters           : {model.count_params():,}")
print(f"Trainable parameters       : {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
print(f"Non-trainable parameters   : {sum([tf.size(w).numpy() for w in model.non_trainable_weights]):,}\n")


# ==========================
# 5. CALLBACKS
# ==========================
callbacks_phase1 = [
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
    ModelCheckpoint("ModelX_efficientnetb4_phase1.h5", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=4, min_lr=1e-6, verbose=1)
]


# ==========================
# 6. TRAINING PHASE 1 — FEATURE EXTRACTION
# ==========================
print("=" * 55)
print("  PHASE 1: FEATURE EXTRACTION (Base Model Frozen)")
print("=" * 55)

history_phase1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks_phase1
)


# ==========================
# 7. FINE-TUNING — PHASE 2
# EfficientNetB4 has ~475 layers; unfreeze last ~100 layers
# ==========================
print("\n" + "=" * 55)
print("  PHASE 2: FINE-TUNING (Partial Unfreeze)")
print("=" * 55)

base_model.trainable = True
fine_tune_at = 375   # Unfreeze from layer 375 onwards (~100 layers)

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

trainable_count = sum([1 for l in base_model.layers if l.trainable])
print(f"Unfrozen base layers        : {trainable_count} layers (from layer {fine_tune_at}+)")
print(f"Remaining frozen base layers: {fine_tune_at} layers\n")

# Recompile with very low learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_phase2 = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
    ModelCheckpoint("chilimodel_efficientnetb4_best.h5", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=5, min_lr=1e-7, verbose=1)
]

history_phase2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks_phase2
)


# ==========================
# 8. SAVE FINAL MODEL
# ==========================
model.save("chilimodel_efficientnetb4_final.h5")
print("\nFinal model saved: chilimodel_efficientnetb4_final.h5")


# ==========================
# 9. COMBINE HISTORY PHASE 1 & PHASE 2 FOR PLOTTING
# ==========================
def combine_history(h1, h2):
    """Combine two training histories"""
    combined = {}
    for key in h1.history:
        combined[key] = h1.history[key] + h2.history[key]
    return combined

history_combined = combine_history(history_phase1, history_phase2)
total_epochs     = len(history_combined['accuracy'])
phase1_epochs    = len(history_phase1.history['accuracy'])


# ==========================
# 10. PLOT TRAINING HISTORY
# ==========================
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history_combined['accuracy'],     label='Train Accuracy', color='blue')
plt.plot(history_combined['val_accuracy'], label='Val Accuracy',   color='orange')
plt.axvline(x=phase1_epochs, color='red', linestyle='--', label=f'Fine-tuning starts (epoch {phase1_epochs})')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_combined['loss'],     label='Train Loss', color='blue')
plt.plot(history_combined['val_loss'], label='Val Loss',   color='orange')
plt.axvline(x=phase1_epochs, color='red', linestyle='--', label=f'Fine-tuning starts (epoch {phase1_epochs})')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("training_history_efficientnetb4.png", bbox_inches='tight', dpi=150)
plt.show()
print("Training history plot saved: training_history_efficientnetb4.png")


# ==========================
# 11. TEST EVALUATION
# ==========================
test_data.reset()
test_loss, test_acc = model.evaluate(test_data, verbose=1)

train_acc_best = max(history_combined['accuracy'])
val_acc_best   = max(history_combined['val_accuracy'])


print("\n" + "=" * 55)
print("  MODEL EVALUATION RESULTS")
print("=" * 55)
print(f"Model               : EfficientNetB4")
print(f"Train Accuracy      : {train_acc_best * 100:.2f}%")
print(f"Validation Accuracy : {val_acc_best   * 100:.2f}%")
print(f"Test Accuracy       : {test_acc        * 100:.2f}%")
print(f"Test Loss           : {test_loss:.4f}")
print("=" * 55)


# ==========================
# 12. CONFUSION MATRIX & CLASSIFICATION REPORT
# ==========================
test_data.reset()
y_pred         = model.predict(test_data)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true         = test_data.classes

print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names,
    cmap="Blues"
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix — EfficientNetB4")
plt.savefig("confusion_matrix_efficientnetb4.png", bbox_inches='tight', dpi=150)
plt.show()
print("Confusion matrix saved: confusion_matrix_efficientnetb4.png")


# ==========================
# 13. GRAD-CAM SETUP
# ==========================
last_conv_layer_name = [
    l.name for l in base_model.layers
    if isinstance(l, tf.keras.layers.Conv2D)
][-1]
print(f"\nGrad-CAM using layer: {last_conv_layer_name}")

grad_model = tf.keras.models.Model(
    inputs=model.input,
    outputs=[
        base_model.get_layer(last_conv_layer_name).output,
        model.output
    ]
)


# ==========================
# 14. GRAD-CAM FUNCTIONS
# ==========================
def make_gradcam_heatmap(img_array, grad_model, pred_index=None):
    """Generate Grad-CAM heatmap for an image"""
    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_tensor, training=False)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads        = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output  = conv_output[0]

    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    max_val = tf.math.reduce_max(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (max_val + 1e-8)
    return heatmap.numpy()


def create_overlay(img_path, heatmap, pred_label, true_label, confidence):
    """Create Grad-CAM overlay on original image"""
    img             = cv2.imread(img_path)
    img             = cv2.resize(img, IMG_SIZE)
    heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    overlay         = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)

    color = (0, 255, 0) if pred_label == true_label else (0, 0, 255)
    cv2.putText(overlay, f"Pred: {pred_label} ({confidence:.2f})",
                (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(overlay, f"True: {true_label}",
                (5, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def save_gradcam_file(img_path, heatmap, pred_label, true_label, confidence, save_dir):
    """Save Grad-CAM overlay image"""
    os.makedirs(save_dir, exist_ok=True)
    overlay_rgb = create_overlay(img_path, heatmap, pred_label, true_label, confidence)
    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
    filename    = os.path.basename(img_path)
    cv2.imwrite(os.path.join(save_dir, filename), overlay_bgr)


# ==========================
# 15. EXECUTE GRAD-CAM (ALL TEST DATA)
# ==========================
base_dir    = "gradcam_results_efficientnetb4"
filepaths   = test_data.filepaths
true_labels = [class_names[i] for i in test_data.classes]

gradcam_results = []
correct = 0
wrong   = 0

print("\nProcessing Grad-CAM for all test data...")

for i, img_path in enumerate(filepaths):
    pred_class = y_pred_classes[i]
    confidence = float(y_pred[i][pred_class])
    label_name = class_names[pred_class]
    true_name  = true_labels[i]

    img       = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    heatmap = make_gradcam_heatmap(img_array, grad_model, pred_index=pred_class)

    if label_name == true_name:
        save_path  = os.path.join(base_dir, "correct", label_name)
        correct   += 1
        is_correct = True
    else:
        save_path  = os.path.join(base_dir, "wrong", f"true_{true_name}_pred_{label_name}")
        wrong     += 1
        is_correct = False

    save_gradcam_file(img_path, heatmap, label_name, true_name, confidence, save_path)

    gradcam_results.append({
        "img_path"  : img_path,
        "heatmap"   : heatmap,
        "pred"      : label_name,
        "true"      : true_name,
        "confidence": confidence,
        "correct"   : is_correct
    })

print(f"\nGrad-CAM processing complete.")
print(f"  Correct predictions : {correct} images → {base_dir}/correct/")
print(f"  Wrong predictions   : {wrong}   images → {base_dir}/wrong/")
print(f"  Total               : {correct + wrong} images")


# ==========================
# 16. VISUALIZE RANDOM GRAD-CAM SAMPLES
# ==========================
samples = random.sample(gradcam_results, min(20, len(gradcam_results)))

fig, axes = plt.subplots(5, 2, figsize=(10, 20))
fig.suptitle("Sample Grad-CAM Results — EfficientNetB4\n(Left: Original Image | Right: Grad-CAM Overlay)",
             fontsize=13, fontweight='bold', y=1.01)

for i, sample in enumerate(samples):
    img_ori = cv2.imread(sample["img_path"])
    img_ori = cv2.resize(img_ori, IMG_SIZE)
    img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)

    axes[i, 0].imshow(img_ori)
    axes[i, 0].set_title(f"True: {sample['true']}", fontsize=10)
    axes[i, 0].axis('off')

    overlay_rgb = create_overlay(
        sample["img_path"],
        sample["heatmap"],
        sample["pred"],
        sample["true"],
        sample["confidence"]
    )

    status = "✓ Correct" if sample["correct"] else "✗ Wrong"
    color  = "green"     if sample["correct"] else "red"

    axes[i, 1].imshow(overlay_rgb)
    axes[i, 1].set_title(
        f"Pred: {sample['pred']} ({sample['confidence']:.2f}) — {status}",
        fontsize=10, color=color
    )
    axes[i, 1].axis('off')

plt.tight_layout()
plt.savefig("gradcam_samples.png", bbox_inches='tight', dpi=150)
plt.show()
print("5-sample Grad-CAM visualization")
