import os

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

from share import *
import config

import random
import threading

import einops
import gradio as gr
import numpy as np
import torch

# === GPU 优化 ===
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


model = create_model('./configs/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./models/control_sd15_scribble.pth', location='cpu'), strict=False)
model = model.cuda()
ddim_sampler = DDIMSampler(model)

GPU_MEMORY_GB = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
MAX_IMAGE_RESOLUTION = 512 if GPU_MEMORY_GB <= 8.5 else 768
MAX_IMAGE_SAMPLES = 2 if GPU_MEMORY_GB <= 8.5 else 8

cancel_event = threading.Event()


def create_canvas():
    return np.full((512, 512, 3), 255, dtype=np.uint8)


def prepare_generation():
    cancel_event.clear()


def stop_generation():
    cancel_event.set()


def raise_if_cancelled():
    if cancel_event.is_set():
        raise gr.Error('生成已停止。')


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    if input_image is None:
        raise gr.Error('请先在涂鸦画板上绘制内容。')
    prompt = prompt.strip() if prompt else ''
    if not prompt:
        raise gr.Error('请输入提示词。')
    num_samples = int(num_samples)
    image_resolution = int(image_resolution)
    if num_samples > MAX_IMAGE_SAMPLES:
        raise gr.Error(f'当前显存下每次最多生成 {MAX_IMAGE_SAMPLES} 张图片。')
    if image_resolution > MAX_IMAGE_RESOLUTION:
        raise gr.Error(f'当前显存下图片分辨率最高为 {MAX_IMAGE_RESOLUTION}。')

    raw_image = np.asarray(input_image)
    if raw_image.ndim == 2:
        raw_image = np.stack([raw_image] * 3, axis=-1)
    if raw_image.ndim == 3 and raw_image.shape[2] == 4:
        raw_image = raw_image[:, :, :3]

    with torch.no_grad():
        img = resize_image(HWC3(raw_image), image_resolution)
        raise_if_cancelled()
        if not np.any(np.min(img, axis=2) < 220):
            raise gr.Error('请先在涂鸦画板上绘制内容。')

        detected_map = np.zeros_like(img, dtype=np.uint8)
        detected_map[np.min(img, axis=2) < 127] = 255

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {
            'c_concat': [control],
            'c_crossattn': [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)],
        }
        un_cond = {
            'c_concat': None if guess_mode else [control],
            'c_crossattn': [model.get_learned_conditioning([n_prompt] * num_samples)],
        }
        shape = (4, img.shape[0] // 8, img.shape[1] // 8)
        raise_if_cancelled()

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        samples, intermediates = ddim_sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond,
            callback=lambda _: raise_if_cancelled(),
        )

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)
        raise_if_cancelled()

        results = []
        for i in range(num_samples):
            x_sample = model.decode_first_stage(samples[i:i + 1])
            x_sample = (einops.rearrange(x_sample, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            results.append(x_sample[0])
    return [255 - detected_map] + results


block = gr.Blocks(title='ControlNet 与用户涂鸦').queue()
with block:
    gr.Markdown('## ControlNet 与用户涂鸦')
    gr.Markdown('Stable Diffusion 1.5 + ControlNet（Scribbles）')
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                value=create_canvas(),
                source='canvas',
                tool='sketch',
                type='numpy',
                image_mode='RGB',
                brush_color='#000000',
                brush_radius=4,
                height=512,
                width=512,
                label='涂鸦画板',
            )
            clear_button = gr.Button('清空画布')
            prompt = gr.Textbox(label='提示词', placeholder='例如：turtle')
            with gr.Row():
                run_button = gr.Button('生成', variant='primary')
                stop_button = gr.Button('停止生成', variant='stop')
            with gr.Accordion('高级选项', open=False):
                num_samples = gr.Slider(label='图片数量', minimum=1, maximum=MAX_IMAGE_SAMPLES, value=1, step=1)
                image_resolution = gr.Slider(label='图片分辨率', minimum=256, maximum=MAX_IMAGE_RESOLUTION, value=512, step=64)
                strength = gr.Slider(label='控制强度', minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                ddim_steps = gr.Slider(label='采样步数', minimum=1, maximum=100, value=20, step=1)
                scale = gr.Slider(label='引导强度', minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                seed = gr.Slider(label='随机种子', minimum=-1, maximum=2147483647, step=1, randomize=True)
                eta = gr.Number(label='eta (DDIM)', value=0.0)
                a_prompt = gr.Textbox(label='正向补充词', value='best quality, extremely detailed')
                n_prompt = gr.Textbox(
                    label='负向提示词',
                    value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality',
                )
        with gr.Column():
            result_gallery = gr.Gallery(
                label='生成结果',
                show_label=False,
                elem_id='gallery',
                columns=2,
                height='auto',
            )

    clear_button.click(fn=create_canvas, inputs=None, outputs=[input_image])
    ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta]
    run_event = run_button.click(
        fn=prepare_generation, inputs=None, outputs=None, queue=False
    ).then(fn=process, inputs=ips, outputs=[result_gallery])
    stop_button.click(fn=stop_generation, inputs=None, outputs=None, queue=False)
    stop_button.click(fn=None, inputs=None, outputs=None, queue=False, cancels=[run_event])


server_port = int(os.environ.get('GRADIO_SERVER_PORT', '7866'))
block.launch(server_name='0.0.0.0', server_port=server_port)
