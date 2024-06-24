# -*- coding: utf-8 -*-
"""
Created on Tue May  9 23:22:57 2023

@author: Xinhao Lan
"""

from dataclasses import dataclass
from PIL import Image
from datasets import load_dataset
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

@dataclass
class TrainingConfig:
    image_size = 256  # the generated image resolution
    train_batch_size = 1
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 12000
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 1
    save_model_epochs = 400
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "output"  # the model name locally and on the HF Hub

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0

class CreateDatasetFromImages(Dataset):
    def __init__(self, csv_path, file_path, resize_height=256, resize_width=256):
        """
        Args:
            csv_path (string): File Parth for csv file
            img_path (string): File path for image file
            transform: transform operation
        """
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.file_path = file_path
        self.to_tensor = transforms.ToTensor()

        self.data_info = pd.read_csv(csv_path, sep = ',')
        self.data_info = self.data_info.fillna(0)
        self.data_info = self.data_info.replace(-1, 1)

        self.image_arr = np.asarray(self.data_info.iloc[1:, 0])  
        self.label_arr = np.asarray(self.data_info.iloc[1:, 14])
        
        self.data_len = len(self.data_info.index) - 1

    def __getitem__(self, index):
        single_image_name = self.image_arr[index]
        img_as_img = Image.open(self.file_path + single_image_name)
        
        # Convert the RGB image to gray image
        # if img_as_img.mode != 'L':
        #     img_as_img = img_as_img.convert('L')
        
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5],[0.5])
        ])
        img_as_img = transform(img_as_img)
        label = self.label_arr[index]
        return (img_as_img, label)
        
    def __len__(self):
        return self.data_len

config = TrainingConfig()
#config.dataset_name = "huggan/smithsonian_butterflies_subset"
#dataset = load_dataset(config.dataset_name, split="train")
#dataset = load_dataset("csv", data_dir = "C:/Users/75581/Desktop/Code/CheXpert-v1.0-small", data_files = 'train_new.csv', split="train")

import matplotlib.pyplot as plt
"""
fig, axs = plt.subplots(1, 4, figsize = (16, 4))
for i, path in enumerate(dataset[:4]['Path']):
    img = Image.open("C:/Users/75581/Desktop/Code/" + path)
    axs[i].imshow(img, cmap = 'gray')
    axs[i].set_axis_off()
fig.show()
"""
from torchvision import transforms
preprocess = transforms.Compose(
    [
     transforms.Resize((config.image_size, config.image_size)),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5]),
     ]
    )

#csv_path = 'D:/CheXpert-v1.0-small/train.csv'
#file_path = 'D:/'

csv_path = "stable-diffusion-main/CheXpert-v1.0-small-sd/train1.csv"
file_path = "stable-diffusion-main/"

dataset = CreateDatasetFromImages(csv_path , file_path, config.image_size, config.image_size)
train_dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config.train_batch_size, 
        shuffle=False,
    )

"""
def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}
dataset.set_transform(transform)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)
"""

from diffusers import UNet2DModel
model = UNet2DModel(
    sample_size = config.image_size,
    in_channels = 3,
    out_channels = 3,
    layers_per_block = 2,
    block_out_channels = (128,128,256,256,512,512),
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
    )

sample_image = dataset[0][0].unsqueeze(0)
print("Input shape:", sample_image.shape)
print("Output shape:", model(sample_image, timestep=0).sample.shape)

from diffusers import DDPMScheduler
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
noise = torch.randn(sample_image.shape)
timesteps = torch.LongTensor([50])
noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)
#print(((noisy_image.permute(0, 2, 3, 1) + 1.0) * 255.5).type(torch.uint8).numpy()[0].shape)
#Image.fromarray(np.reshape(((noisy_image.permute(0, 2, 3, 1) + 1.0) * 255.5).type(torch.uint8).numpy()[0], (-1, 256)), mode = 'L').show()

import torch.nn.functional as F
noise_pred = model(noisy_image, timesteps).sample
loss = F.mse_loss(noise_pred, noise)

from diffusers.optimization import get_cosine_schedule_with_warmup

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

from diffusers import DDPMPipeline
import math
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("L", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid

def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
    ).images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")

from accelerate import Accelerator
from tqdm.auto import tqdm
from pathlib import Path
def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        logging_dir=os.path.join(config.output_dir, "logs"),
    )
    
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")
    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0
    
    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        print('Epoch', epoch, '!!!!!!')
        for step, batch in enumerate(train_dataloader):
            clean_images = batch[0]
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline.save_pretrained(config.output_dir)
        
"""
from accelerate import notebook_launcher
args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
notebook_launcher(train_loop, args, num_processes=1)
"""
train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
import glob

sample_images = sorted(glob.glob(f"{config.output_dir}/samples/*.png"))
Image.open(sample_images[-1]).show()
