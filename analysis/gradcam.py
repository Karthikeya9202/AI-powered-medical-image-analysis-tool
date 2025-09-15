import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

cnn_model = ResNet50(weights="imagenet")

def generate_gradcam(image_bgr):
    """Generate Grad-CAM heatmap for medical image."""
    img_resized = cv2.resize(image_bgr, (224, 224))
    x = np.expand_dims(img_resized, axis=0)
    x = preprocess_input(x)

    grad_model = tf.keras.models.Model(
        [cnn_model.inputs],
        [cnn_model.get_layer("conv5_block3_out").output, cnn_model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    heatmap = cv2.resize(np.array(heatmap), (image_bgr.shape[1], image_bgr.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_bgr, 0.6, heatmap_color, 0.4, 0)

    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
