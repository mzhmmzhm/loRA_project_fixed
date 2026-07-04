from datasets import load_dataset, concatenate_datasets, Image
import torchvision.transforms as transforms
import torch
import os
from PIL import Image as PILImage

preprocess = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def preprocess_images(examples):
    processed_images = torch.stack([preprocess(image.convert("RGB")) for image in examples["image"]])
    return {"image": processed_images}

def load_and_preprocess_data(Dataset1_dir: str, Dataset2_dir: str):
    # I used two datasets for this project; users can use them as they wish.
    # Load the "Dataset1" from the specified directory using the imagefolder format.
    print(f"Loading Dataset1 from {Dataset1_dir}...")
    Dataset1 = load_dataset("imagefolder", data_dir=Dataset1_dir)
    print("Dataset1 loaded.")

    # Load the "Dataset2" from the specified directory using the imagefolder format.
    print(f"Loading Dataset2 from {Dataset2_dir}...")
    Dataset2 = load_dataset("imagefolder", data_dir=Dataset2_dir)
    print("Dataset2 loaded.")


    # Concatenate the training splits of the two datasets into a single dataset.
    # This creates a combined dataset for training.
    print("Concatenating datasets")
    combined_dataset = concatenate_datasets([Dataset1["train"], Dataset2["train"]])
    print(f"Combined dataset: {combined_dataset}")

    # Apply the preprocessing function to the combined dataset.
    # `batched=True` allows processing images in batches for efficiency.
    # `remove_columns=["image"]` removes the original image column to save memory.
    print("Preprocessing images...")
    preprocessed_dataset = combined_dataset.map(preprocess_images, batched=True, remove_columns=["image"])
    print("Image preprocessing complete.")

    # Display the preprocessed dataset information.
    # Note: In a standalone script, display() might not work directly.
    # You might use print(preprocessed_dataset) instead.
    # display(preprocessed_dataset)

    return preprocessed_dataset

class CaptionedImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.images = []
        self.captions = []
        for fname in sorted(os.listdir(data_dir)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(data_dir, fname)
                txt_path = os.path.join(data_dir, os.path.splitext(fname)[0] + '.txt')
                self.images.append(img_path)
                if os.path.exists(txt_path):
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        self.captions.append(f.read().strip())
                else:
                    self.captions.append("")
        print(f"Loaded {len(self.images)} images with captions")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = PILImage.open(self.images[idx]).convert("RGB")
        image = preprocess(image)
        return {"image": image, "caption": self.captions[idx]}

def load_captioned_dataset(Dataset1_dir, Dataset2_dir=None):
    dataset1 = CaptionedImageDataset(Dataset1_dir)
    if Dataset2_dir:
        dataset2 = CaptionedImageDataset(Dataset2_dir)
        from torch.utils.data import ConcatDataset
        combined = ConcatDataset([dataset1, dataset2])
        print(f"Combined dataset: {len(combined)} samples")
        return combined
    return dataset1

if __name__ == "__main__":
    Dataset1_dir = "path/to/your/Dataset1"
    Dataset2_dir = "path/to/your/Dataset2"

    print("Demonstrating data loading and preprocessing function call:")
    preprocessed_data = load_and_preprocess_data(Dataset1_dir, Dataset2_dir)
    print(preprocessed_data)
    