import os
import random
import threading
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL, ControlNetModel, StableDiffusionXLControlNetPipeline

from prompt_translation import enhance_prompt_for_generation, translate_prompt_to_english


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

ROOT = Path(__file__).resolve().parent
SDXL_ROOT = ROOT / "models" / "sdxl"
BASE_MODEL_DIR = SDXL_ROOT / "stable-diffusion-xl-base-1.0"
VAE_DIR = SDXL_ROOT / "vae-fp16-fix"
SCRIBBLE_DIR = SDXL_ROOT / "controlnet-scribble-sdxl-1.0"
CANNY_DIR = SDXL_ROOT / "controlnet-canny-sdxl-1.0"

MAX_IMAGE_RESOLUTION = 1024
# SDXL is trained at 1024; below that the model tends to tile the composition.
DEFAULT_IMAGE_RESOLUTION = 1024
OFFLOAD_MODE = os.environ.get("SDXL_OFFLOAD", "sequential").strip().lower()
cancel_event = threading.Event()


def _has_controlnet(model_dir, filename):
    return (model_dir / "config.json").is_file() and (model_dir / filename).is_file()


def select_controlnet():
    if _has_controlnet(SCRIBBLE_DIR, "diffusion_pytorch_model.safetensors"):
        return SCRIBBLE_DIR, "scribble"
    if _has_controlnet(CANNY_DIR, "diffusion_pytorch_model.fp16.safetensors"):
        return CANNY_DIR, "canny"
    raise SystemExit(
        "找不到 SDXL ControlNet 权重。请先运行: "
        "python scripts\\download_sdxl_scribble.py"
    )


CONTROLNET_DIR, CONTROLNET_MODE = select_controlnet()


def create_canvas():
    return np.full((DEFAULT_IMAGE_RESOLUTION, DEFAULT_IMAGE_RESOLUTION, 3), 255, dtype=np.uint8)


def clamp_image_size(image, max_side):
    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side > max_side:
        scale = max_side / longest_side
        height = int(np.round(height * scale))
        width = int(np.round(width * scale))
    height = max(256, int(np.round(height / 64.0)) * 64)
    width = max(256, int(np.round(width / 64.0)) * 64)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def prepare_generation():
    cancel_event.clear()


def stop_generation():
    cancel_event.set()


def raise_if_cancelled():
    if cancel_event.is_set():
        raise gr.Error('生成已停止。')


def _resize_control_image(image, target_height, target_width):
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def _build_control_image(image, target_height, target_width):
    resized = _resize_control_image(image, target_height, target_width)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    detected_map = np.zeros_like(gray, dtype=np.uint8)
    detected_map[gray < 127] = 255

    if CONTROLNET_MODE == "scribble":
        # Scribble ControlNet expects dark strokes over a light background.
        sketch = 255 - detected_map
        control = np.stack([sketch] * 3, axis=-1)
    else:
        # Canny ControlNet expects white edges over a black background.
        edges = cv2.Canny(gray, 100, 200)
        edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)
        control = np.stack([edges] * 3, axis=-1)
    return detected_map, control


def _load_pipeline():
    controlnet = ControlNetModel.from_pretrained(
        str(CONTROLNET_DIR),
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    vae = AutoencoderKL.from_pretrained(
        str(VAE_DIR),
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
        str(BASE_MODEL_DIR),
        controlnet=controlnet,
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        local_files_only=True,
        add_watermarker=False,
    )
    if OFFLOAD_MODE == "model":
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.enable_sequential_cpu_offload()
    pipeline.vae.enable_slicing()
    pipeline.vae.enable_tiling()
    return pipeline


pipe = _load_pipeline()


def process(input_image, prompt, a_prompt, n_prompt, image_resolution, ddim_steps, strength, scale, prompt_weight, seed):
    prepare_generation()
    if input_image is None:
        raise gr.Error('请先在涂鸦画板上绘制内容。')
    prompt = prompt.strip() if prompt else ''
    if not prompt:
        raise gr.Error('请输入提示词。')

    translated_prompt = translate_prompt_to_english(prompt) or prompt
    prompt_for_model = enhance_prompt_for_generation(prompt, translated_prompt)
    condition_text = ', '.join(part.strip() for part in [prompt_for_model, a_prompt] if part and part.strip())
    effective_scale = min(20.0, max(0.1, float(scale) * float(prompt_weight)))

    image_resolution = int(image_resolution)
    if image_resolution > MAX_IMAGE_RESOLUTION:
        raise gr.Error(f'当前 SDXL 模式最高使用 {MAX_IMAGE_RESOLUTION} 分辨率。')

    raw_image = np.asarray(input_image)
    if raw_image.ndim == 2:
        raw_image = np.stack([raw_image] * 3, axis=-1)
    if raw_image.ndim == 3 and raw_image.shape[2] == 4:
        raw_image = raw_image[:, :, :3]
    raw_image = np.ascontiguousarray(raw_image)

    img = clamp_image_size(raw_image, image_resolution)
    if not np.any(np.min(img, axis=2) < 220):
        raise gr.Error('请先在涂鸦画板上绘制内容。')

    height, width = img.shape[:2]
    detected_map, control = _build_control_image(img, height, width)
    control_image = Image.fromarray(control.astype(np.uint8))

    if seed == -1:
        seed = random.randint(0, 2147483647)
    generator_device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(int(seed))

    def callback_on_step_end(_pipe, _step, _timestep, callback_kwargs):
        raise_if_cancelled()
        return callback_kwargs

    try:
        with torch.inference_mode():
            result = pipe(
                prompt=condition_text,
                negative_prompt=n_prompt,
                image=control_image,
                height=height,
                width=width,
                num_inference_steps=int(ddim_steps),
                guidance_scale=effective_scale,
                controlnet_conditioning_scale=float(strength),
                generator=generator,
                num_images_per_prompt=1,
                callback_on_step_end=callback_on_step_end,
                output_type="pil",
            ).images[0]
    except RuntimeError as error:
        if 'cancelled' in str(error).lower() or cancel_event.is_set():
            raise gr.Error('生成已停止。')
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return [255 - detected_map, np.asarray(result)]


block = gr.Blocks(title='ControlNet SDXL 与用户涂鸦').queue()
with block:
    gr.Markdown('## ControlNet SDXL 与用户涂鸦')
    gr.Markdown(f'Stable Diffusion XL + ControlNet（{CONTROLNET_MODE}）')
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
                height=768,
                width=768,
                label='涂鸦画板',
            )
            clear_button = gr.Button('清空画布')
            prompt = gr.Textbox(label='提示词', placeholder='例如：广州塔 / Canton Tower')
            with gr.Row():
                run_button = gr.Button('生成', variant='primary')
                stop_button = gr.Button('停止生成', variant='stop')
            with gr.Accordion('高级选项', open=False):
                image_resolution = gr.Slider(label='图片分辨率', minimum=512, maximum=MAX_IMAGE_RESOLUTION, value=DEFAULT_IMAGE_RESOLUTION, step=64)
                strength = gr.Slider(label='控制强度', minimum=0.0, maximum=2.0, value=0.9, step=0.05)
                ddim_steps = gr.Slider(label='采样步数', minimum=4, maximum=60, value=20, step=1)
                scale = gr.Slider(label='引导强度', minimum=0.1, maximum=20.0, value=6.0, step=0.1)
                prompt_weight = gr.Slider(
                    label='提示词权重',
                    minimum=1.0,
                    maximum=2.0,
                    value=1.2,
                    step=0.05,
                )
                seed = gr.Slider(label='随机种子', minimum=-1, maximum=2147483647, step=1, randomize=True)
                a_prompt = gr.Textbox(label='正向补充词', value='best quality, detailed, sharp focus')
                n_prompt = gr.Textbox(
                    label='负向提示词',
                    value='lowres, bad anatomy, bad hands, missing fingers, extra digit, cropped, worst quality, low quality, blurry, watermark, text',
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
    ips = [input_image, prompt, a_prompt, n_prompt, image_resolution, ddim_steps, strength, scale, prompt_weight, seed]
    run_event = run_button.click(
        fn=prepare_generation, inputs=None, outputs=None, queue=False
    ).then(fn=process, inputs=ips, outputs=[result_gallery])
    stop_button.click(fn=stop_generation, inputs=None, outputs=None, queue=False)
    stop_button.click(fn=None, inputs=None, outputs=None, queue=False, cancels=[run_event])


def main():
    server_port = int(os.environ.get('GRADIO_SERVER_PORT', '7867'))
    block.launch(server_name='0.0.0.0', server_port=server_port)


if __name__ == '__main__':
    main()
