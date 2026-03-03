<h1 align="center">SmartPix: Unified Image Generation</h1>

<h4 align="center">
    <p>
        <a href="#1-overview">Overview</a> |
        <a href="#2-what-can-smartpix-do">Capabilities</a> |
        <a href="#3-quick-start">Quick Start</a> |
        <a href="#4-finetune">Finetune</a> |
        <a href="#license">License</a>
    <p>
</h4>



## 1. Overview

SmartPix is a unified image generation model that can generate a wide range of images from multi-modal prompts. It is designed to be simple, flexible, and easy to use. We provide [inference code](#3-quick-start) so that everyone can explore more functionalities of SmartPix.

Existing image generation models often require loading several additional network modules (such as ControlNet, IP-Adapter, Reference-Net, etc.) and performing extra preprocessing steps (e.g., face detection, pose estimation, cropping, etc.) to generate a satisfactory image. However, **we believe that the future image generation paradigm should be more simple and flexible, that is, generating various images directly through arbitrarily multi-modal instructions without the need for additional plugins and operations, similar to how GPT works in language generation.** 

Due to the limited resources, SmartPix still has room for improvement. We will continue to optimize it, and hope it inspires more universal image-generation models. You can also easily fine-tune SmartPix without worrying about designing networks for specific tasks; you just need to prepare the corresponding data, and then run the [script](#4-finetune). Imagination is no longer limited; everyone can construct any image-generation task, and perhaps we can achieve very interesting, wonderful, and creative things.

If you have any questions, ideas, or interesting tasks you want SmartPix to accomplish, feel free to discuss with us: 2906698981@qq.com, wangyueze@tju.edu.cn, zhengliu1026@gmail.com. We welcome any feedback to help us improve the model.


## 2. What Can SmartPix do?

SmartPix is a unified image generation model that you can use to perform various tasks, including but not limited to text-to-image generation, subject-driven generation, Identity-Preserving Generation, image editing, and image-conditioned generation. **SmartPix doesn't need additional plugins or operations, it can automatically identify the features (e.g., required object, human pose, depth mapping) in input images according to the text prompt.**
We showcase some examples in [inference.ipynb](inference.ipynb). And in [inference_demo.ipynb](inference_demo.ipynb), we show an interesting pipeline to generate and modify an image.

Here is the illustrations of SmartPix's capabilities: 
- You can control the image generation flexibly via SmartPix
![demo](./imgs/demo_cases.png)
- Referring Expression Generation: You can input multiple images and use simple, general language to refer to the objects within those images. SmartPix can automatically recognize the necessary objects in each image and generate new images based on them. No additional operations, such as image cropping or face detection, are required.
![demo](./imgs/referring.png)

If you are not entirely satisfied with certain functionalities or wish to add new capabilities, you can try [fine-tuning SmartPix](#4-finetune).



## 3. Quick Start


### Using SmartPix
Install via Github:
```bash
git clone https://github.com/Durai-Kannan/SmartPix.git
cd SmartPix
pip install -e .
```

You also can create a new environment to avoid conflicts:
```bash
# Create a python 3.10.13 conda env (you could also use virtualenv)
conda create -n smartpix python=3.10.13
conda activate smartpix

# Install pytorch with your CUDA version, e.g.
pip install torch==2.3.1+cu118 torchvision --extra-index-url https://download.pytorch.org/whl/cu118

git clone https://github.com/Durai-Kannan/SmartPix.git
cd SmartPix
pip install -e .
```

Here are some examples:
```python
from SmartPix import SmartPixPipeline

pipe = SmartPixPipeline.from_pretrained("Shitao/OmniGen-v1")  
# Note: Your local model path is also acceptable, such as 'pipe = SmartPixPipeline.from_pretrained(your_local_model_path)', where all files in your_local_model_path should be organized as https://huggingface.co/Shitao/OmniGen-v1/tree/main

## Text to Image
images = pipe(
    prompt="A curly-haired man in a red shirt is drinking tea.", 
    height=1024, 
    width=1024, 
    guidance_scale=2.5,
    seed=0,
)
images[0].save("example_t2i.png")  # save output PIL Image

## Multi-modal to Image
# In the prompt, we use the placeholder to represent the image. The image placeholder should be in the format of <img><|image_*|></img>
# You can add multiple images in the input_images. Please ensure that each image has its placeholder. For example, for the list input_images [img1_path, img2_path], the prompt needs to have two placeholders: <img><|image_1|></img>, <img><|image_2|></img>.
images = pipe(
    prompt="A man in a black shirt is reading a book. The man is the right man in <img><|image_1|></img>.",
    input_images=["./imgs/test_cases/two_man.jpg"],
    height=1024, 
    width=1024,
    guidance_scale=2.5, 
    img_guidance_scale=1.6,
    seed=0
)
images[0].save("example_ti2i.png")  # save output PIL image
```
- If out of memory, you can set `offload_model=True`. If the inference time is too long when inputting multiple images, you can reduce the `max_input_image_size`.  For the required resources and the method to run SmartPix efficiently, please refer to [docs/inference.md#requiremented-resources](docs/inference.md#requiremented-resources).
- For more examples of image generation, you can refer to [inference.ipynb](inference.ipynb) and [inference_demo.ipynb](inference_demo.ipynb)
- For more details about the argument in inference, please refer to [docs/inference.md](docs/inference.md). 


### Using Diffusers

[Diffusers docs](https://huggingface.co/docs/diffusers/main/en/using-diffusers/omnigen)


### Gradio Demo

We construct an online demo in [Huggingface](https://huggingface.co/spaces/Shitao/OmniGen).

For the local gradio demo, you need to install `pip install gradio spaces`, and then you can run:
```python
pip install gradio spaces
python app.py
```

#### Use Google Colab
To use with Google Colab, please use the following command:

```
!git clone https://github.com/Durai-Kannan/SmartPix.git
%cd SmartPix
!pip install -e .
!pip install gradio spaces
!python app.py --share
```

## 4. Finetune
We provide a training script `train.py` to fine-tune SmartPix. 
Here is a toy example about LoRA finetune:
```bash
accelerate launch --num_processes=1 train.py \
    --model_name_or_path Shitao/OmniGen-v1 \
    --batch_size_per_device 2 \
    --condition_dropout_prob 0.01 \
    --lr 1e-3 \
    --use_lora \
    --lora_rank 8 \
    --json_file ./toy_data/toy_subject_data.jsonl \
    --image_path ./toy_data/images \
    --max_input_length_limit 18000 \
    --keep_raw_resolution \
    --max_image_size 1024 \
    --gradient_accumulation_steps 1 \
    --ckpt_every 10 \
    --epochs 200 \
    --log_every 1 \
    --results_dir ./results/toy_finetune_lora
```

Please refer to [docs/fine-tuning.md](docs/fine-tuning.md) for more details (e.g. full finetune).

### Contributors:
Thank all our contributors for their efforts and warmly welcome new members to join in!

<a href="https://github.com/Durai-Kannan/SmartPix/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Durai-Kannan/SmartPix" />
</a>

## License
This repo is licensed under the [MIT License](LICENSE). 
