import os
import cv2
import tensorflow as tf
from typing import Tuple

IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
IMAGENET_STD = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)


def get_train_augmentation() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(factor=10.0 / 360.0),
            tf.keras.layers.RandomContrast(0.2),
            tf.keras.layers.RandomBrightness(0.2),
        ],
        name="data_augmentation",
    )


def preprocess_image(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Resizes, normalizes and standardizes image for transfer learning."""
    image = tf.image.resize(image, [224, 224])

    image = image / 255.0

    image = (image - IMAGENET_MEAN) / IMAGENET_STD

    return image, label


def load_dataset(
    data_dir: str, batch_size: int = 32, is_training: bool = True
) -> tf.data.Dataset:
    if not os.path.exists(data_dir):
        print(
            f"Warning: Data directory {data_dir} does not exist. Please setup dataset."
        )
        return tf.data.Dataset.from_tensor_slices([])

    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="categorical",
        batch_size=batch_size,
        image_size=(224, 224),
        shuffle=is_training,
    )

    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        aug_model = get_train_augmentation()
        dataset = dataset.map(
            lambda x, y: (aug_model(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return dataset.prefetch(tf.data.AUTOTUNE)


def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    event_time_sec: float,
    class_name: str,
    window_sec: float = 1.0,
) -> None:
    if not os.path.exists(video_path):
        print(f"Video {video_path} not found.")
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    start_frame = int(max(0, (event_time_sec - window_sec)) * fps)
    end_frame = int((event_time_sec + window_sec) * fps)

    class_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame

    video_id = os.path.basename(video_path).split(".")[0]

    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        sample_step = max(1, int(fps / 5))
        if current_frame % sample_step == 0:
            frame_filename = os.path.join(
                class_dir, f"{video_id}_t{event_time_sec}_f{current_frame}.jpg"
            )
            cv2.imwrite(frame_filename, frame)

        current_frame += 1

    cap.release()
