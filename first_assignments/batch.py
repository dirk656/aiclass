import os

import cv2
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


raw_data = [
    ("imagedata/n0/n1021.jpg", "n0"),   
    ("imagedata/n0/n1022.jpg", "n0"),   
    ("imagedata/n1/大象.jpg", "n1"),    
    ("imagedata/n2/老虎.jpg", "n2"),    
]

def batch_loader(raw_data: list[tuple[str, str]], batch_size: int) -> DataLoader:
    records: list[tuple[torch.Tensor, str, str]] = []

    for path, label in raw_data:
        img = cv2.imread(path)
      
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        image_name = os.path.basename(path)
        records.append((tensor, label, image_name))

    return DataLoader(records, batch_size=batch_size, shuffle=False)


def show_batch(batch_tensor: torch.Tensor, batch_labels: list[str], batch_names: list[str], batch_id: int) -> None:
    plt.figure(figsize=(10, 4))
    for i in range(batch_tensor.shape[0]):
        img = batch_tensor[i].permute(1, 2, 0).numpy()
        plt.subplot(1, batch_tensor.shape[0], i + 1)
        plt.imshow(img)
        plt.title(f"{batch_labels[i]}: {batch_names[i]}", fontsize=9)
        plt.axis("off")
    plt.suptitle(f"Batch {batch_id}")
    plt.tight_layout()
    plt.show()


def main() -> None:
    batch_size = 2
    data_loader = batch_loader(raw_data, batch_size)

    for i, (batch_tensor, batch_labels, batch_names) in enumerate(data_loader, start=1):
        print(f"Batch {i} -> shape: {tuple(batch_tensor.shape)}")
        print(f"labels: {list(batch_labels)}")
        print(f"names: {list(batch_names)}")
        show_batch(batch_tensor, list(batch_labels), list(batch_names), i)


if __name__ == "__main__":
    main()