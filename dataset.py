import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms() -> T.Compose:
    return T.Compose(
        [
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.RandomResizedCrop(224, scale=(0.8, 1.0)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_val_transforms() -> T.Compose:
    return T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class FootballDataset(Dataset):
    """PyTorch Dataset for football action images."""

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: T.Compose = None,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def load_dataset(
    data_dir: str,
    batch_size: int = 32,
    is_training: bool = True,
) -> DataLoader:
    """
    Build a DataLoader from a directory of class-labelled sub-folders.
    Expects structure:  data_dir/<class_name>/<image_files>
    """
    if not os.path.exists(data_dir):
        print(
            f"Warning: Data directory {data_dir} does not exist. Please setup dataset."
        )
        return DataLoader(FootballDataset([], [], None))

    transform = get_train_transforms() if is_training else get_val_transforms()

    image_paths: List[str] = []
    labels: List[int] = []
    class_names = sorted(
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    )
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        for fname in os.listdir(class_dir):
            if os.path.splitext(fname)[1].lower() in valid_ext:
                image_paths.append(os.path.join(class_dir, fname))
                labels.append(class_to_idx[class_name])

    dataset = FootballDataset(image_paths, labels, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )


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
