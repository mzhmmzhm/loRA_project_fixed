import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL, get_scheduler
def train_lora_diffusion(model_id: str, preprocessed_dataset, lora_config: LoraConfig, training_args: TrainingArguments):
    """
    Args:
        model_id (str): The ID of the base Stable Diffusion model.
        preprocessed_dataset: The dataset containing preprocessed images.
        lora_config (LoraConfig): The LoRA configuration object.
        training_args (TrainingArguments): The training arguments object.
    """
    # 1. Initialize Accelerator
    # Handles device placement, mixed precision, and gradient accumulation.
    accelerator = Accelerator(
        mixed_precision=training_args.mixed_precision if hasattr(training_args, 'mixed_precision') else None,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps if hasattr(training_args, 'gradient_accumulation_steps') else 1,
    )

    # 2. Load the VAE and Text Encoder models
    # VAE for encoding/decoding images to/from latent space.
    # Text Encoder for conditioning the diffusion process (even with dummy data for unlabelled images).
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")

    # 3. Load the pre-trained diffusion model's UNet component and apply LoRA
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch.float32)

    # Apply the LoRA configuration to the UNet model.
    lora_model = get_peft_model(unet, lora_config)
    lora_model.print_trainable_parameters()

    # 4. Define optimizer and learning rate scheduler
    optimizer = optim.AdamW(lora_model.parameters(), lr=training_args.learning_rate)

    # Create DataLoader for the dataset.
    train_dataloader = DataLoader(
        preprocessed_dataset,
        batch_size=training_args.per_device_train_batch_size,
        shuffle=True,
        num_workers=training_args.dataloader_num_workers if hasattr(training_args, 'dataloader_workers') else 0,
    )

    # Define the learning rate scheduler.
    lr_scheduler = get_scheduler(
        training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.get_warmup_steps(training_args.num_train_epochs * len(train_dataloader)),
        num_training_steps=training_args.num_train_epochs * len(train_dataloader),
    )

    # Prepare models, optimizer, dataloader, and scheduler with Accelerator.
    lora_model, vae, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        lora_model, vae, text_encoder, optimizer, train_dataloader, lr_scheduler
    )

    # Get the model's dtype (should be float32 after fixing UNet loading)
    model_dtype = next(lora_model.parameters()).dtype
    # Keep VAE in its native precision (float32) for numerical stability
    vae.to(dtype=torch.float32)
    # Get the text encoder's dtype.
    text_encoder_dtype = next(text_encoder.parameters()).dtype

    # 5. Define the noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # 6. Set the models to training or evaluation mode
    lora_model.train()
    vae.eval() # VAE is typically kept in evaluation mode.
    text_encoder.eval() # Text encoder is typically kept in evaluation mode.

    # 7. Get the number of training steps
    num_update_steps_per_epoch = len(train_dataloader)
    if training_args.gradient_accumulation_steps is not None:
        num_update_steps_per_epoch = num_update_steps_per_epoch // training_args.gradient_accumulation_steps
    num_training_steps = training_args.num_train_epochs * num_update_steps_per_epoch

    print(f"Starting training for {training_args.num_train_epochs} epochs")

    # 8. Training loop 
    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)

    global_step = 0

    for epoch in range(training_args.num_train_epochs):
        lora_model.train()
        vae.eval()
        text_encoder.eval()

        try:
            for step, batch in enumerate(train_dataloader):
                batch = {k: v.to(accelerator.device) if hasattr(v, "to") and isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                images = batch["image"]

                # Ensure images have a batch dimension and are tensors (handling potential issues from DataLoader)
                if isinstance(images, list) and len(images) > 0 and isinstance(images[0], list):
                    images = [torch.tensor(img).to(accelerator.device) for img in images]
                    images = torch.stack(images, dim=0)
                elif isinstance(images, list) and len(images) > 0 and isinstance(images[0], torch.Tensor):
                     images = torch.stack(images, dim=0)

                if isinstance(images, torch.Tensor):
                    if images.ndim == 2:
                        images = images.unsqueeze(0).unsqueeze(0)
                    elif images.ndim == 3:
                        images = images.unsqueeze(0)
                    elif images.ndim == 4:
                         pass
                    else:
                         print(f"Warning: Unexpected image tensor dimensions: {images.ndim}")
                         continue
                else:
                     print(f"Warning: Images variable is not a tensor: {type(images)}")
                     continue

                images = images.to(dtype=model_dtype)

                # Encode images to latents using VAE
                latents = vae.encode(images).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                latents = latents.to(dtype=model_dtype)

                # Diffusion Model Training Specifics
                noise = torch.randn_like(latents, dtype=model_dtype)
                bs = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps).to(dtype=model_dtype)

                captions = batch["caption"]  

                # Tokenize the captions
                text_inputs = tokenizer(
                    captions,
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                    return_tensors="pt"
                ).to(accelerator.device)

                # Encode with the text encoder
                with torch.no_grad():  # Text encoder stays frozen
                    encoder_hidden_states = text_encoder(
                        text_inputs.input_ids
                    )[0].to(lora_model.dtype)
                
                # Predict the noise residual and calculate loss with gradient accumulation
                with accelerator.accumulate(lora_model):
                    model_output = lora_model(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states.to(lora_model.dtype)
                    ).sample

                    loss = torch.nn.functional.mse_loss(model_output.float(), noise.float(), reduction="mean")

                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        if training_args.max_grad_norm is not None:
                            accelerator.unscale_gradients()
                            torch.nn.utils.clip_grad_norm_(lora_model.parameters(), training_args.max_grad_norm)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                # Update progress bar and log
                if accelerator.sync_gradients:
                     progress_bar.update(1)
                     global_step += 1
                     if accelerator.is_local_main_process and global_step % training_args.logging_steps == 0:
                         print(f"Epoch {epoch+1}, Step {step}/{len(train_dataloader)} - Global Step {global_step} - Loss: {loss.item()}")
                else:
                     if accelerator.is_local_main_process and step % training_args.logging_steps == 0:
                         print(f"Epoch {epoch+1}, Step {step}/{len(train_dataloader)} - Accumulating Loss: {loss.item()}")


            # Save checkpoints periodically
            if accelerator.is_local_main_process and accelerator.sync_gradients and training_args.save_steps > 0 and global_step % training_args.save_steps == 0 and global_step != 0:
                unwrapped_model = accelerator.unwrap_model(lora_model)
                output_dir = f"{training_args.output_dir}/checkpoint-{global_step}"
                unwrapped_model.save_pretrained(output_dir)

        except Exception as e:
            print(f"An error occurred during training epoch {epoch}: {type(e).__name__} - {e}")
            break # Break the epoch loop if an error occurs

    print("Training finished.")

    # Save the final model
    if accelerator.is_local_main_process:
        try:
            unwrapped_model = accelerator.unwrap_model(lora_model)
            unwrapped_model.save_pretrained(training_args.output_dir)
            print(f"Final model saved to {training_args.output_dir}")
        except Exception as e:
            print(f"An error occurred while saving the final model: {type(e).__name__} - {e}")
