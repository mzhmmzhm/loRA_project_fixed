# LoRA Fine-Tuning for Stable Diffusion 2.1

A modular and memory-efficient pipeline for fine-tuning **Stable Diffusion v2.1** using LoRA (Low-Rank Adaptation) adapters on your custom datasets. Developed in Google Colab, but portable to your local system.

##  Project Highlights

- **Base Model:** Stable Diffusion 2.1 via Hugging Face Diffusers
- **Parameter-Efficient**: Utilizes PEFT/LoRA for low-resource fine-tuning
- **Multi-Dataset Support:** Train on one or many datasets; user-configurable
- **Modular Codebase:** Clear separation for preprocessing, training, and inference
- **Colab & Local:** Ready for GPU/Colab or local run (with version and memory notes)

##  Folder Structure
```
LoRA_Project/
├── data_utils.py
├── train.py
├── inference.py
├── requirements.txt
├── .gitignore
├── README.md
```


##  Installation

1. **Clone the repo:**
    ```
    git clone https://github.com/Keshavaaa/LoRA_Project.git
    cd LoRA_Project
    ```

2. **(Optional but recommended) Create a virtual environment:**
    ```bash
    python -m venv loraenv
    source loraenv/bin/activate    # or .\loraenv\Scripts\activate (Windows)
    ```

3. **Install dependencies:**
    ```
    pip install -r requirements.txt
    ```
    - *Colab users*: You can also copy the pip install lines directly into your Colab code cell.
    - *Local users*: Make sure PyTorch is installed with the correct CUDA version for your GPU.

##  Data Preparation

- Organize your images using the [imagefolder format](https://huggingface.co/docs/datasets/image_load#imagefolder).
- Use one or multiple datasets; simply configure paths in `data_utils.py`.

Sample folder structure:
```
data/
├── dataset1/
│ ├── img1.jpg
│ └── ...
└── dataset2/
├── picA.png
└── ...
```

##  Data Preprocessing

```
from data_utils import load_and_preprocess_data

# Example (replace with your data directories):
dataset1_dir = "path/to/your/dataset1"
dataset2_dir = "path/to/your/dataset2"

preprocessed = load_and_preprocess_data(dataset1_dir, dataset2_dir)
# Or call with just one directory if you wish
##  Training

from peft import LoraConfig
from transformers import TrainingArguments
from train import train_lora_diffusion

model_id = "runwayml/stable-diffusion-v1-5"
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["to_q", "to_k", "to_v"],
    lora_dropout=0.01,
)

training_args = TrainingArguments(
    output_dir="./lora_training_output",
    num_train_epochs=2,
    per_device_train_batch_size=1,
    learning_rate=1e-4,
    save_steps=50,
    logging_steps=100,
)

# Assuming your preprocessed dataset is named 'preprocessed'
train_lora_diffusion(model_id, preprocessed, lora_config, training_args)
```

All LoRA/tuning settings are user-configurable.

Trained/checkpointed models are saved in output_dir.

##  Inference

```
from inference import run_inference

image = run_inference(
    model_id="runwayml/stable-diffusion-v1-5",
    lora_path="./lora_training_output",
    prompt="A lake at dawn in impressionist style",
    negative_prompt="blurry, cartoon, animation",
    guidance_scale=7.5,
)
image.show()  # Or image.save("result.png")
```
## Environment & Version Notes

Colab-tested versions (see requirements.txt):
```
text
accelerate==0.30.1
peft==0.11.1
bitsandbytes==0.43.1
transformers==4.41.2
diffusers==0.29.0

```
## Challenges Faced & Lessons Learned

Throughout the development of this LoRA fine-tuning pipeline, I encountered and resolved several real-world machine learning engineering challenges. These technical and practical hurdles are common in deep learning workflows and their solutions are valuable for both reproducibility and learning.

1. **Data Loader & Batch Format Issues**
   * **Problem**: Hugging Face Datasets sometimes yielded image batches as lists or nested lists, instead of batched tensors.

   * **Solution**: Added custom preprocess_images and batch-stacking logic in data_utils.py to guarantee all batches are single, properly shaped tensors before entering the model.

2. **Device and Dtype Mismatches**
   * **Problem**: Frequent errors like Input type (c10::Half) and bias type (float) should be the same occurred due to mixed precision (float16 vs float32) and CPU/GPU placement mismatches, especially with VAE and UNet.

   * **Solution**: Standardized all tensors and models to use .to(accelerator.device).to(model_dtype) everywhere in the training loop, and always cast VAE weights after Accelerator preparation.

3. **VAE Input and Dimensionality**
   * **Problem**: The autoencoder VAE requires input in [batch, channels, height, width]. Sometimes images arrived shaped [channels, height, width] (missing batch) or [batch, height, width] (missing channel).

   * **Solution**: Inserted controlled unsqueeze logic and thorough tensor shape checks in the batch preprocessing step before passing to VAE.

4. **Training on Small Datasets**
   * **Problem**: Only ~100 images, causing overfitting and limiting the capacity of LoRA to meaningfully adapt the model.

   * **Solution**: Tuned the LoRA r rank and lora_alpha to lower values, increased dropout, and augmented data. Added prompt engineering for more distinct control tokens, and monitored outputs closely for signs of memorization.

5. **Version & Dependency Drift**
   * **Problem**: Some packages (diffusers, transformers, peft, accelerate) updated rapidly—code that worked in Colab could break locally or months later.

   * **Solution**: Captured exact working versions in requirements.txt, documented Colab-specific idiosyncrasies, and provided environment warnings in the README.

6. **Colab vs Local Environment Issues**
   * **Problem**: Certain installs (bitsandbytes, CUDA) required extra setup locally, while Colab “just worked.”

   * **Solution**: Added detailed notes for users to install the correct Torch and CUDA versions, explained bitsandbytes as optional, and summarized common errors and solutions in the README.

8. **Other Lessons Learned**
   * **Checkpoint Handling**: Saved LoRA checkpoints more frequently to avoid progress loss from Colab crashes.

   * **Importing and File Management**: Used .gitignore to prevent accidental upload of checkpoints, data, or outputs.

## Summary:
By solving these challenges, I built a modular, and easy-to-adapt LoRA fine-tuning pipeline for Stable Diffusion, suitable for Colab and local use. The experience covers not just model training, but also practical engineering, reproducibility, and the documentation skills needed for real open-source collaboration.
