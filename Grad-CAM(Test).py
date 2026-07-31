import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import h5py
import json
import shutil


# CONFIG
# ---------------------
model_path  = "FINALMODEL/Chiliefficientnetb4_finalmodel.h5"
image_input = "cercosy.jpg"
# your model path and test image path here

IMG_SIZE = (380, 380)

def fix_h5_quantization(src_path):
    with h5py.File(src_path, 'r') as f:
        model_config_raw = f.attrs.get('model_config')
        if isinstance(model_config_raw, bytes):
            model_config_raw = model_config_raw.decode('utf-8')

    config_dict = json.loads(model_config_raw)

    def remove_key(d, key_to_remove):
        if isinstance(d, dict):
            d.pop(key_to_remove, None)
            for v in d.values():
                remove_key(v, key_to_remove)
        elif isinstance(d, list):
            for item in d:
                remove_key(item, key_to_remove)

    remove_key(config_dict, 'quantization_config')

    fixed_path = src_path.replace('.h5', '_fixed.h5')
    shutil.copy2(src_path, fixed_path)

    with h5py.File(fixed_path, 'a') as f:
        f.attrs['model_config'] = json.dumps(config_dict)

    print(f"  ✓ Model diperbaiki → {fixed_path}")
    return fixed_path


# ============================================================
# 1. LOAD MODEL
# ============================================================
print("=" * 60)
print("  LOADING MODEL ...")
print("=" * 60)

fixed_model_path = fix_h5_quantization(model_path)
model = tf.keras.models.load_model(fixed_model_path)
print(f"  ✓ Model berhasil dimuat!\n")

num_classes = model.output_shape[-1]

# Detect class names from dataset folder structure
_possible_dirs = [
    "CHILIDISEASEDATASET/test",
    "CHILIDISEASEDATASET/train",
    "CHILIDISEASEDATASET/val",
]
class_names = None
for _d in _possible_dirs:
    if os.path.isdir(_d):
        _candidates = sorted([
            n for n in os.listdir(_d)
            if os.path.isdir(os.path.join(_d, n))
        ])
        if len(_candidates) == num_classes:
            class_names = _candidates
            break

if class_names is None:
    class_names = [f"Kelas_{i}" for i in range(num_classes)]

print(f"  Number of classes : {num_classes}")
print(f"  Class names   : {class_names}\n")


# ============================================================
# 2. LOAD & PREPROCESS 
# ============================================================
img_pil   = tf.keras.preprocessing.image.load_img(image_input, target_size=IMG_SIZE)
img_array = tf.keras.preprocessing.image.img_to_array(img_pil)
img_batch = np.expand_dims(img_array, axis=0)

img_cv    = cv2.imread(image_input)
img_cv    = cv2.resize(img_cv, IMG_SIZE)
img_rgb   = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)


# ============================================================
# 3. PREDICT
# ============================================================
predictions = model.predict(img_batch, verbose=0)[0]
pred_idx    = int(np.argmax(predictions))
pred_label  = class_names[pred_idx]
pred_conf   = float(predictions[pred_idx])

print("=" * 60)
print("  RESULTS")
print("=" * 60)
for i, (name, score) in enumerate(zip(class_names, predictions)):
    marker = " ◄ PREDICTION" if i == pred_idx else ""
    print(f"  {name:<30} {score * 100:>9.2f}%{marker}")
print(f"\n  Prediction : {pred_label}  ({pred_conf * 100:.2f}%)\n")


# ============================================================
# 4. GRAD-CAM
# ============================================================

# Cari layer Conv2D terakhir dan indeks base_model di dalam model utama
base_model = None
base_model_idx = -1
for i, layer in enumerate(model.layers):
    if isinstance(layer, tf.keras.Model):
        base_model = layer
        base_model_idx = i
        break

if base_model is None:
    raise ValueError("EfficientNet model is not found.")

last_conv_name = [
    l.name for l in base_model.layers
    if isinstance(l, tf.keras.layers.Conv2D)
][-1]
print(f"  Grad-CAM layer : {last_conv_name}\n")


# 1. Buat base_model baru yang mengekstrak feature map SEKALIGUS output aslinya
new_base_model = tf.keras.Model(
    inputs=base_model.input,
    outputs=[base_model.get_layer(last_conv_name).output, base_model.output]
)

# 2. Mulai membangun graph dari input model utama
inputs = model.input
x_conv, x = new_base_model(inputs)

# 3. Lanjutkan koneksi ke sisa layer setelah base_model (GAP, Dense, dll)
for layer in model.layers[base_model_idx + 1:]:
    x = layer(x)

# 4. Buat grad_model dari graph yang sudah terhubung secara eksplisit
grad_model = tf.keras.Model(inputs=inputs, outputs=[x_conv, x])


def compute_gradcam(img_batch, grad_model, pred_idx):
    img_tensor = tf.cast(img_batch, tf.float32)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_tensor, training=False)
        loss = preds[:, pred_idx]
    grads        = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out     = conv_out[0]
    heatmap      = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap)
    heatmap      = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

heatmap = compute_gradcam(img_batch, grad_model, pred_idx)

# Overlay heatmap ke gambar asli
heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
heatmap_uint8   = np.uint8(255 * heatmap_resized)
heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
overlay_bgr     = cv2.addWeighted(img_cv, 0.55, heatmap_colored, 0.45, 0)
overlay_rgb     = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)


#-----------------
# Visualization
bar_colors = ["#1e8a4e" if i == pred_idx else "#95a5a6" for i in range(num_classes)]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor("#1e1e2e")

axes[0].imshow(img_rgb)
axes[0].set_title("Gambar Input", fontsize=13, fontweight="bold", color="white", pad=10)
axes[0].axis("off")

short_names = [n.replace("_", "\n") for n in class_names]
bars = axes[1].barh(short_names, predictions * 100, color=bar_colors,
                    edgecolor="white", linewidth=0.5)
axes[1].set_xlim(0, 110)
axes[1].set_xlabel("Probability (%)", color="white", fontsize=11)
axes[1].set_title("Classification Score per Class", fontsize=13,
                  fontweight="bold", color="white", pad=10)
axes[1].tick_params(colors="white")
axes[1].spines[:].set_color("#444466")
axes[1].set_facecolor("#2a2a3e")

for bar, score in zip(bars, predictions):
    axes[1].text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                 f"{score * 100:.1f}%", va="center", ha="left",
                 color="white", fontsize=10, fontweight="bold")

patch_pred  = mpatches.Patch(color="#1e8a4e", label="Highest Prediction")
patch_other = mpatches.Patch(color="#95a5a6", label="Other")
axes[1].legend(handles=[patch_pred, patch_other],
               loc="lower right", facecolor="#2a2a3e",
               labelcolor="white", fontsize=9)

axes[2].imshow(overlay_rgb)
axes[2].set_title(
    f"Grad-CAM  →  {pred_label}\nConfidence: {pred_conf * 100:.2f}%",
    fontsize=13, fontweight="bold", color="white", pad=10
)
axes[2].axis("off")

sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(0, 1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes[2], fraction=0.046, pad=0.04)
cbar.set_label("Aktivasi (low → high)", color="white", fontsize=9)
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

plt.suptitle(
    f"Chili Disease Classification Results ·  Model: EfficientNetB4",
    fontsize=13, fontweight="bold", color="white", y=1
)
plt.tight_layout()

save_dir  = os.path.dirname(model_path)
save_path = os.path.join(save_dir, "hasil_pengujian_" +
                         os.path.splitext(os.path.basename(image_input))[0] + ".png")
plt.savefig(save_path, bbox_inches="tight", dpi=150, facecolor=fig.get_facecolor())
plt.show()

print("=" * 60)
print("  RINGKASAN")
print("=" * 60)
print(f"  Test Image     : {os.path.basename(image_input)}")
print(f"  Prediction       : {pred_label}")
print(f"  Confidence     : {pred_conf * 100:.2f}%")
print(f"  Visualization    : {save_path}")
print("=" * 60)