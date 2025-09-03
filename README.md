---
title: 🧩 DiffuseCraft
emoji: 🧩🖼️
colorFrom: red
colorTo: pink
sdk: gradio
sdk_version: 5.44.1
app_file: app.py
pinned: true
license: mit
short_description: Stunning images using stable diffusion.
preload_from_hub:
  - madebyollin/sdxl-vae-fp16-fix config.json,diffusion_pytorch_model.safetensors
---

# DiffuseCraft

## Overview
**DiffuseCraft** is a versatile image generation tool to perform a variety of tasks, including generating new images, inpainting, upscaling, and analyzing PNG info. It allows fine-grained control over models, prompts, and advanced image generation parameters.

| Description | Link |
| ----------- | ---- |
| 📙 Colab Notebook | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/R3gm/DiffuseCraft/blob/main/DiffuseCraft_Colab.ipynb) |
| 🎉 Repository | [![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-black?style=flat-square&logo=github)](https://github.com/R3gm/DiffuseCraft) |
| 🚀 Online DEMO | [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/r3gm/DiffuseCraft) |

## Key Functions
1. **Dowload models**  
   Download Checkpoints and LoRAs from Hugging Face, Civitai or Google Drive
2. **Image Generation**  
   Generate images from text prompts using advanced models. Includes support for:
   - **Prompt & Negative Prompt** – control the desired content and unwanted elements.  
   - **Model Selection** – e.g., `Laxhar/noobai-XL-1.1`.  
   - **LoRA, VAE, IP-Adapter** – optional tools for enhanced control.

3. **Inpainting**  
   Edit parts of images by providing masks to modify specific areas while keeping the rest intact.

4. **PNG Info & Preprocessing**  
   Read PNG images parameters and apply preprocessing before further generation tasks.

5. **Upscaler**  
   Improve image resolution while preserving details.

6. **Advanced Controls**  
   - **Steps, CFG, Sampler, Schedule Type, Image Size, Seed, Pag scale, Freeu, clip skip** – customize generation parameters.  
   - **Face Restoration, ControlNet, Styles, Detail Fix** – optional enhancements.

## Summary Table

| Function                   | Description |
|----------------------------|-------------|
| **Text-to-Image (txt2img)**| Generate images from prompts with fine-grained control over style, details, and model parameters. |
| **Inpainting**              | Modify specific areas of an image using masks. |
| **Upscaler**                | Enhance image resolution and clarity. |
| **PNG Info & Preprocessing**| Read and preprocess PNG images for generation or editing. |
| **LoRA & VAE Models**       | Apply optional models to influence generation. |
| **Face Restoration & Styles**| Enhance faces or apply stylistic changes to images. |
| **Download Output**         | Save generated or processed images. |
