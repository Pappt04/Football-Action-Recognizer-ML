import tensorflow as tf


def build_action_recognizer(
    input_shape: tuple[int, int, int] = (224, 224, 3), num_classes: int = 6
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """
    Builds and returns the EfficientNetB0 transfer learning model.

    Architecture:
        Input → EfficientNetB0 (frozen) → GlobalAvgPool → Dense(256) → Dropout(0.5) → Dense(6)

    Notes:
        - include_preprocessing=False because ImageNet mean/std normalization
          is applied externally in the tf.data pipeline.
        - Returns (full_model, backbone) so the backbone can be partially
          unfrozen for fine-tuning phase 2.

    Args:
        input_shape: (H, W, C) - defaults to (224, 224, 3)
        num_classes: number of output classes (default 6)

    Returns:
        model:    full Keras Model ready for compilation
        backbone: the EfficientNetB0 sub-model (for fine-tuning control)
    """
    inputs = tf.keras.Input(shape=input_shape, name="input")

    # ── Backbone: EfficientNetB0 pretrained on ImageNet ────────────────────
    # include_preprocessing=False: we handle normalization in the data pipeline
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        include_preprocessing=False,
    )
    backbone.trainable = False  # Frozen during Phase 1

    x = backbone(inputs, training=False)

    # ── Custom classification head ──────────────────────────────────────────
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="dense_256")(x)
    x = tf.keras.layers.Dropout(0.5, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="football_action_recognizer")
    return model, backbone
