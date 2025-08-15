import spaces
import os
from stablepy import (
    Model_Diffusers,
    SCHEDULE_TYPE_OPTIONS,
    SCHEDULE_PREDICTION_TYPE_OPTIONS,
    check_scheduler_compatibility,
    TASK_AND_PREPROCESSORS,
    FACE_RESTORATION_MODELS,
    scheduler_names,
)
from constants import (
    DIRECTORY_MODELS,
    DIRECTORY_LORAS,
    DIRECTORY_VAES,
    DIRECTORY_EMBEDS,
    DIRECTORY_UPSCALERS,
    DOWNLOAD_MODEL,
    DOWNLOAD_VAE,
    DOWNLOAD_LORA,
    LOAD_DIFFUSERS_FORMAT_MODEL,
    DIFFUSERS_FORMAT_LORAS,
    DOWNLOAD_EMBEDS,
    CIVITAI_API_KEY,
    HF_TOKEN,
    TASK_STABLEPY,
    TASK_MODEL_LIST,
    UPSCALER_DICT_GUI,
    UPSCALER_KEYS,
    PROMPT_W_OPTIONS,
    WARNING_MSG_VAE,
    SDXL_TASK,
    MODEL_TYPE_TASK,
    POST_PROCESSING_SAMPLER,
    SUBTITLE_GUI,
    HELP_GUI,
    EXAMPLES_GUI_HELP,
    EXAMPLES_GUI,
    RESOURCES,
    DIFFUSERS_CONTROLNET_MODEL,
    IP_MODELS,
    MODE_IP_OPTIONS,
)
from stablepy.diffusers_vanilla.style_prompt_config import STYLE_NAMES
import torch
import re
import time
from PIL import ImageFile
from utils import (
    download_things,
    get_model_list,
    extract_parameters,
    get_my_lora,
    get_model_type,
    extract_exif_data,
    create_mask_now,
    download_diffuser_repo,
    get_used_storage_gb,
    delete_model,
    progress_step_bar,
    html_template_message,
    escape_html,
)
from image_processor import preprocessor_tab
from datetime import datetime
import gradio as gr
import logging
import diffusers
import warnings
from stablepy import logger
from diffusers import FluxPipeline
# import urllib.parse
import subprocess

subprocess.run("rm -rf /data-nvme/zerogpu-offload/*", env={}, shell=True)
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.backends.cuda.matmul.allow_tf32 = True
# os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
print(os.getenv("SPACES_ZERO_GPU"))

directories = [DIRECTORY_MODELS, DIRECTORY_LORAS, DIRECTORY_VAES, DIRECTORY_EMBEDS, DIRECTORY_UPSCALERS]
for directory in directories:
    os.makedirs(directory, exist_ok=True)

# Download stuffs
for url in [url.strip() for url in DOWNLOAD_MODEL.split(',')]:
    if not os.path.exists(f"./models/{url.split('/')[-1]}"):
        download_things(DIRECTORY_MODELS, url, HF_TOKEN, CIVITAI_API_KEY)
for url in [url.strip() for url in DOWNLOAD_VAE.split(',')]:
    if not os.path.exists(f"./vaes/{url.split('/')[-1]}"):
        download_things(DIRECTORY_VAES, url, HF_TOKEN, CIVITAI_API_KEY)
for url in [url.strip() for url in DOWNLOAD_LORA.split(',')]:
    if not os.path.exists(f"./loras/{url.split('/')[-1]}"):
        download_things(DIRECTORY_LORAS, url, HF_TOKEN, CIVITAI_API_KEY)

# Download Embeddings
for url_embed in DOWNLOAD_EMBEDS:
    if not os.path.exists(f"./embedings/{url_embed.split('/')[-1]}"):
        download_things(DIRECTORY_EMBEDS, url_embed, HF_TOKEN, CIVITAI_API_KEY)

# Build list models
embed_list = get_model_list(DIRECTORY_EMBEDS)
embed_list = [
    (os.path.splitext(os.path.basename(emb))[0], emb) for emb in embed_list
]
single_file_model_list = get_model_list(DIRECTORY_MODELS)
model_list = LOAD_DIFFUSERS_FORMAT_MODEL + single_file_model_list
lora_model_list = get_model_list(DIRECTORY_LORAS)
lora_model_list.insert(0, "None")
lora_model_list = lora_model_list + DIFFUSERS_FORMAT_LORAS
vae_model_list = get_model_list(DIRECTORY_VAES)
vae_model_list.insert(0, "BakedVAE")
vae_model_list.insert(0, "None")

print('\033[33m🏁 Download and listing of valid models completed.\033[0m')

flux_repo = "camenduru/FLUX.1-dev-diffusers"
flux_pipe = FluxPipeline.from_pretrained(
    flux_repo,
    transformer=None,
    torch_dtype=torch.bfloat16,
).to("cuda")
components = flux_pipe.components
delete_model(flux_repo)
# components = None

#######################
# GUI
#######################
logging.getLogger("diffusers").setLevel(logging.ERROR)
diffusers.utils.logging.set_verbosity(40)
warnings.filterwarnings(action="ignore", category=FutureWarning, module="diffusers")
warnings.filterwarnings(action="ignore", category=UserWarning, module="diffusers")
warnings.filterwarnings(action="ignore", category=FutureWarning, module="transformers")
logger.setLevel(logging.DEBUG)

CSS = """
.contain { display: flex; flex-direction: column; }
#component-0 { height: 100%; }
#gallery { flex-grow: 1; }
#load_model { height: 50px; }
"""


class GuiSD:
    def __init__(self, stream=True):
        self.model = None
        self.status_loading = False
        self.sleep_loading = 4
        self.last_load = datetime.now()
        self.inventory = []

    def update_storage_models(self, storage_floor_gb=24, required_inventory_for_purge=3):
        while get_used_storage_gb() > storage_floor_gb:
            if len(self.inventory) < required_inventory_for_purge:
                break
            removal_candidate = self.inventory.pop(0)
            delete_model(removal_candidate)

    def update_inventory(self, model_name):
        if model_name not in single_file_model_list:
            self.inventory = [
                m for m in self.inventory if m != model_name
            ] + [model_name]
        print(self.inventory)

    def load_new_model(self, model_name, vae_model, task, controlnet_model, progress=gr.Progress(track_tqdm=True)):

        # download link model > model_name

        self.update_storage_models()

        vae_model = vae_model if vae_model != "None" else None
        model_type = get_model_type(model_name)
        dtype_model = torch.bfloat16 if model_type == "FLUX" else torch.float16

        if not os.path.exists(model_name):
            _ = download_diffuser_repo(
                repo_name=model_name,
                model_type=model_type,
                revision="main",
                token=True,
            )

        self.update_inventory(model_name)

        for i in range(68):
            if not self.status_loading:
                self.status_loading = True
                if i > 0:
                    time.sleep(self.sleep_loading)
                    print("Previous model ops...")
                break
            time.sleep(0.5)
            print(f"Waiting queue {i}")
            yield "Waiting queue"

        self.status_loading = True

        yield f"Loading model: {model_name}"

        if vae_model == "BakedVAE":
            vae_model = model_name
        elif vae_model:
            vae_type = "SDXL" if "sdxl" in vae_model.lower() else "SD 1.5"
            if model_type != vae_type:
                gr.Warning(WARNING_MSG_VAE)

        print("Loading model...")

        try:
            start_time = time.time()

            if self.model is None:
                self.model = Model_Diffusers(
                    base_model_id=model_name,
                    task_name=TASK_STABLEPY[task],
                    vae_model=vae_model,
                    type_model_precision=dtype_model,
                    retain_task_model_in_cache=False,
                    controlnet_model=controlnet_model,
                    device="cpu",
                    env_components=components,
                )
                self.model.advanced_params(image_preprocessor_cuda_active=True)
            else:
                if self.model.base_model_id != model_name:
                    load_now_time = datetime.now()
                    elapsed_time = max((load_now_time - self.last_load).total_seconds(), 0)

                    if elapsed_time <= 9:
                        print("Waiting for the previous model's time ops...")
                        time.sleep(9 - elapsed_time)

                self.model.device = torch.device("cpu")
                self.model.load_pipe(
                    model_name,
                    task_name=TASK_STABLEPY[task],
                    vae_model=vae_model,
                    type_model_precision=dtype_model,
                    retain_task_model_in_cache=False,
                    controlnet_model=controlnet_model,
                )

            end_time = time.time()
            self.sleep_loading = max(min(int(end_time - start_time), 10), 4)
        except Exception as e:
            self.last_load = datetime.now()
            self.status_loading = False
            self.sleep_loading = 4
            raise e

        self.last_load = datetime.now()
        self.status_loading = False

        yield f"Model loaded: {model_name}"

    # @spaces.GPU(duration=59)
    @torch.inference_mode()
    def generate_pipeline(
        self,
        prompt,
        neg_prompt,
        num_images,
        steps,
        cfg,
        clip_skip,
        seed,
        lora1,
        lora_scale1,
        lora2,
        lora_scale2,
        lora3,
        lora_scale3,
        lora4,
        lora_scale4,
        lora5,
        lora_scale5,
        lora6,
        lora_scale6,
        lora7,
        lora_scale7,
        sampler,
        schedule_type,
        schedule_prediction_type,
        img_height,
        img_width,
        model_name,
        vae_model,
        task,
        image_control,
        preprocessor_name,
        preprocess_resolution,
        image_resolution,
        style_prompt,  # list []
        style_json_file,
        image_mask,
        strength,
        low_threshold,
        high_threshold,
        value_threshold,
        distance_threshold,
        recolor_gamma_correction,
        tile_blur_sigma,
        controlnet_output_scaling_in_unet,
        controlnet_start_threshold,
        controlnet_stop_threshold,
        textual_inversion,
        syntax_weights,
        upscaler_model_path,
        upscaler_increases_size,
        upscaler_tile_size,
        upscaler_tile_overlap,
        hires_steps,
        hires_denoising_strength,
        hires_sampler,
        hires_prompt,
        hires_negative_prompt,
        hires_before_adetailer,
        hires_after_adetailer,
        hires_schedule_type,
        hires_guidance_scale,
        controlnet_model,
        loop_generation,
        leave_progress_bar,
        disable_progress_bar,
        image_previews,
        display_images,
        save_generated_images,
        filename_pattern,
        image_storage_location,
        retain_compel_previous_load,
        retain_detailfix_model_previous_load,
        retain_hires_model_previous_load,
        t2i_adapter_preprocessor,
        t2i_adapter_conditioning_scale,
        t2i_adapter_conditioning_factor,
        xformers_memory_efficient_attention,
        freeu,
        generator_in_cpu,
        adetailer_inpaint_only,
        adetailer_verbose,
        adetailer_sampler,
        adetailer_active_a,
        prompt_ad_a,
        negative_prompt_ad_a,
        strength_ad_a,
        face_detector_ad_a,
        person_detector_ad_a,
        hand_detector_ad_a,
        mask_dilation_a,
        mask_blur_a,
        mask_padding_a,
        adetailer_active_b,
        prompt_ad_b,
        negative_prompt_ad_b,
        strength_ad_b,
        face_detector_ad_b,
        person_detector_ad_b,
        hand_detector_ad_b,
        mask_dilation_b,
        mask_blur_b,
        mask_padding_b,
        retain_task_cache_gui,
        guidance_rescale,
        image_ip1,
        mask_ip1,
        model_ip1,
        mode_ip1,
        scale_ip1,
        image_ip2,
        mask_ip2,
        model_ip2,
        mode_ip2,
        scale_ip2,
        pag_scale,
        face_restoration_model,
        face_restoration_visibility,
        face_restoration_weight,
    ):
        info_state = html_template_message("Navigating latent space...")
        yield info_state, gr.update(), gr.update()

        vae_model = vae_model if vae_model != "None" else None
        loras_list = [lora1, lora2, lora3, lora4, lora5, lora6, lora7]
        vae_msg = f"VAE: {vae_model}" if vae_model else ""
        msg_lora = ""

        print("Config model:", model_name, vae_model, loras_list)

        task = TASK_STABLEPY[task]

        params_ip_img = []
        params_ip_msk = []
        params_ip_model = []
        params_ip_mode = []
        params_ip_scale = []

        all_adapters = [
            (image_ip1, mask_ip1, model_ip1, mode_ip1, scale_ip1),
            (image_ip2, mask_ip2, model_ip2, mode_ip2, scale_ip2),
        ]

        if not hasattr(self.model.pipe, "transformer"):
            for imgip, mskip, modelip, modeip, scaleip in all_adapters:
                if imgip:
                    params_ip_img.append(imgip)
                    if mskip:
                        params_ip_msk.append(mskip)
                    params_ip_model.append(modelip)
                    params_ip_mode.append(modeip)
                    params_ip_scale.append(scaleip)

        concurrency = 5
        self.model.stream_config(concurrency=concurrency, latent_resize_by=1, vae_decoding=False)

        if task != "txt2img" and not image_control:
            raise ValueError("Reference image is required. Please upload one in 'Image ControlNet/Inpaint/Img2img'.")

        if task in ["inpaint", "repaint"] and not image_mask:
            raise ValueError("Mask image not found. Upload one in 'Image Mask' to proceed.")

        if "https://" not in str(UPSCALER_DICT_GUI[upscaler_model_path]):
            upscaler_model = upscaler_model_path
        else:
            url_upscaler = UPSCALER_DICT_GUI[upscaler_model_path]

            if not os.path.exists(f"./{DIRECTORY_UPSCALERS}/{url_upscaler.split('/')[-1]}"):
                download_things(DIRECTORY_UPSCALERS, url_upscaler, HF_TOKEN)

            upscaler_model = f"./{DIRECTORY_UPSCALERS}/{url_upscaler.split('/')[-1]}"

        logging.getLogger("ultralytics").setLevel(logging.INFO if adetailer_verbose else logging.ERROR)

        adetailer_params_A = {
            "face_detector_ad": face_detector_ad_a,
            "person_detector_ad": person_detector_ad_a,
            "hand_detector_ad": hand_detector_ad_a,
            "prompt": prompt_ad_a,
            "negative_prompt": negative_prompt_ad_a,
            "strength": strength_ad_a,
            # "image_list_task" : None,
            "mask_dilation": mask_dilation_a,
            "mask_blur": mask_blur_a,
            "mask_padding": mask_padding_a,
            "inpaint_only": adetailer_inpaint_only,
            "sampler": adetailer_sampler,
        }

        adetailer_params_B = {
            "face_detector_ad": face_detector_ad_b,
            "person_detector_ad": person_detector_ad_b,
            "hand_detector_ad": hand_detector_ad_b,
            "prompt": prompt_ad_b,
            "negative_prompt": negative_prompt_ad_b,
            "strength": strength_ad_b,
            # "image_list_task" : None,
            "mask_dilation": mask_dilation_b,
            "mask_blur": mask_blur_b,
            "mask_padding": mask_padding_b,
        }
        pipe_params = {
            "prompt": prompt,
            "negative_prompt": neg_prompt,
            "img_height": img_height,
            "img_width": img_width,
            "num_images": num_images,
            "num_steps": steps,
            "guidance_scale": cfg,
            "clip_skip": clip_skip,
            "pag_scale": float(pag_scale),
            "seed": seed,
            "image": image_control,
            "preprocessor_name": preprocessor_name,
            "preprocess_resolution": preprocess_resolution,
            "image_resolution": image_resolution,
            "style_prompt": style_prompt if style_prompt else "",
            "style_json_file": "",
            "image_mask": image_mask,  # only for Inpaint
            "strength": strength,  # only for Inpaint or ...
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
            "value_threshold": value_threshold,
            "distance_threshold": distance_threshold,
            "recolor_gamma_correction": float(recolor_gamma_correction),
            "tile_blur_sigma": int(tile_blur_sigma),
            "lora_A": lora1 if lora1 != "None" else None,
            "lora_scale_A": lora_scale1,
            "lora_B": lora2 if lora2 != "None" else None,
            "lora_scale_B": lora_scale2,
            "lora_C": lora3 if lora3 != "None" else None,
            "lora_scale_C": lora_scale3,
            "lora_D": lora4 if lora4 != "None" else None,
            "lora_scale_D": lora_scale4,
            "lora_E": lora5 if lora5 != "None" else None,
            "lora_scale_E": lora_scale5,
            "lora_F": lora6 if lora6 != "None" else None,
            "lora_scale_F": lora_scale6,
            "lora_G": lora7 if lora7 != "None" else None,
            "lora_scale_G": lora_scale7,
            "textual_inversion": embed_list if textual_inversion else [],
            "syntax_weights": syntax_weights,  # "Classic"
            "sampler": sampler,
            "schedule_type": schedule_type,
            "schedule_prediction_type": schedule_prediction_type,
            "xformers_memory_efficient_attention": xformers_memory_efficient_attention,
            "gui_active": True,
            "loop_generation": loop_generation,
            "controlnet_conditioning_scale": float(controlnet_output_scaling_in_unet),
            "control_guidance_start": float(controlnet_start_threshold),
            "control_guidance_end": float(controlnet_stop_threshold),
            "generator_in_cpu": generator_in_cpu,
            "FreeU": freeu,
            "adetailer_A": adetailer_active_a,
            "adetailer_A_params": adetailer_params_A,
            "adetailer_B": adetailer_active_b,
            "adetailer_B_params": adetailer_params_B,
            "leave_progress_bar": leave_progress_bar,
            "disable_progress_bar": disable_progress_bar,
            "image_previews": image_previews,
            "display_images": display_images,
            "save_generated_images": save_generated_images,
            "filename_pattern": filename_pattern,
            "image_storage_location": image_storage_location,
            "retain_compel_previous_load": retain_compel_previous_load,
            "retain_detailfix_model_previous_load": retain_detailfix_model_previous_load,
            "retain_hires_model_previous_load": retain_hires_model_previous_load,
            "t2i_adapter_preprocessor": t2i_adapter_preprocessor,
            "t2i_adapter_conditioning_scale": float(t2i_adapter_conditioning_scale),
            "t2i_adapter_conditioning_factor": float(t2i_adapter_conditioning_factor),
            "upscaler_model_path": upscaler_model,
            "upscaler_increases_size": upscaler_increases_size,
            "upscaler_tile_size": upscaler_tile_size,
            "upscaler_tile_overlap": upscaler_tile_overlap,
            "hires_steps": hires_steps,
            "hires_denoising_strength": hires_denoising_strength,
            "hires_prompt": hires_prompt,
            "hires_negative_prompt": hires_negative_prompt,
            "hires_sampler": hires_sampler,
            "hires_before_adetailer": hires_before_adetailer,
            "hires_after_adetailer": hires_after_adetailer,
            "hires_schedule_type": hires_schedule_type,
            "hires_guidance_scale": hires_guidance_scale,
            "ip_adapter_image": params_ip_img,
            "ip_adapter_mask": params_ip_msk,
            "ip_adapter_model": params_ip_model,
            "ip_adapter_mode": params_ip_mode,
            "ip_adapter_scale": params_ip_scale,
            "face_restoration_model": face_restoration_model,
            "face_restoration_visibility": face_restoration_visibility,
            "face_restoration_weight": face_restoration_weight,
        }

        # kwargs for diffusers pipeline
        if guidance_rescale:
            pipe_params["guidance_rescale"] = guidance_rescale

        self.model.device = torch.device("cuda:0")
        if hasattr(self.model.pipe, "transformer") and loras_list != ["None"] * self.model.num_loras:
            self.model.pipe.transformer.to(self.model.device)
            print("transformer to cuda")

        actual_progress = 0
        info_images = gr.update()
        for img, [seed, image_path, metadata] in self.model(**pipe_params):
            info_state = progress_step_bar(actual_progress, steps)
            actual_progress += concurrency
            if image_path:
                info_images = f"Seeds: {str(seed)}"
                if vae_msg:
                    info_images = info_images + "<br>" + vae_msg

                if "Cannot copy out of meta tensor; no data!" in self.model.last_lora_error:
                    msg_ram = "Unable to process the LoRAs due to high RAM usage; please try again later."
                    print(msg_ram)
                    msg_lora += f"<br>{msg_ram}"

                for status, lora in zip(self.model.lora_status, self.model.lora_memory):
                    if status:
                        msg_lora += f"<br>Loaded: {lora}"
                    elif status is not None:
                        msg_lora += f"<br>Error with: {lora}"

                if msg_lora:
                    info_images += msg_lora

                info_images = info_images + "<br>" + "GENERATION DATA:<br>" + escape_html(metadata[-1]) + "<br>-------<br>"

                download_links = "<br>".join(
                    [
                        f'<a href="{path.replace("/images/", "/file=/home/user/app/images/")}" download="{os.path.basename(path)}">Download Image {i + 1}</a>'
                        for i, path in enumerate(image_path)
                    ]
                )
                if save_generated_images:
                    info_images += f"<br>{download_links}"

                info_state = "COMPLETE"

            yield info_state, img, info_images


def dynamic_gpu_duration(func, duration, *args):

    # @torch.inference_mode()
    @spaces.GPU(duration=duration)
    def wrapped_func():
        yield from func(*args)

    return wrapped_func()


@spaces.GPU
def dummy_gpu():
    return None


def sd_gen_generate_pipeline(*args):
    gpu_duration_arg = int(args[-1]) if args[-1] else 59
    verbose_arg = int(args[-2])
    load_lora_cpu = args[-3]
    generation_args = args[:-3]
    lora_list = [
        None if item == "None" else item
        for item in [args[7], args[9], args[11], args[13], args[15], args[17], args[19]]
    ]
    lora_status = [None] * sd_gen.model.num_loras

    msg_load_lora = "Updating LoRAs in GPU..."
    if load_lora_cpu:
        msg_load_lora = "Updating LoRAs in CPU..."

    if lora_list != sd_gen.model.lora_memory and lora_list != [None] * sd_gen.model.num_loras:
        yield msg_load_lora, gr.update(), gr.update()

    # Load lora in CPU
    if load_lora_cpu:
        lora_status = sd_gen.model.load_lora_on_the_fly(
            lora_A=lora_list[0], lora_scale_A=args[8],
            lora_B=lora_list[1], lora_scale_B=args[10],
            lora_C=lora_list[2], lora_scale_C=args[12],
            lora_D=lora_list[3], lora_scale_D=args[14],
            lora_E=lora_list[4], lora_scale_E=args[16],
            lora_F=lora_list[5], lora_scale_F=args[18],
            lora_G=lora_list[6], lora_scale_G=args[20],
        )
        print(lora_status)

    sampler_name = args[21]
    schedule_type_name = args[22]
    _, _, msg_sampler = check_scheduler_compatibility(
        sd_gen.model.class_name, sampler_name, schedule_type_name
    )
    if msg_sampler:
        gr.Warning(msg_sampler)

    if verbose_arg:
        for status, lora in zip(lora_status, lora_list):
            if status:
                gr.Info(f"LoRA loaded in CPU: {lora}")
            elif status is not None:
                gr.Warning(f"Failed to load LoRA: {lora}")

        if lora_status == [None] * sd_gen.model.num_loras and sd_gen.model.lora_memory != [None] * sd_gen.model.num_loras and load_lora_cpu:
            lora_cache_msg = ", ".join(
                str(x) for x in sd_gen.model.lora_memory if x is not None
            )
            gr.Info(f"LoRAs in cache: {lora_cache_msg}")

    msg_request = f"Requesting {gpu_duration_arg}s. of GPU time.\nModel: {sd_gen.model.base_model_id}"
    if verbose_arg:
        gr.Info(msg_request)
        print(msg_request)
    yield msg_request.replace("\n", "<br>"), gr.update(), gr.update()

    start_time = time.time()

    # yield from sd_gen.generate_pipeline(*generation_args)
    yield from dynamic_gpu_duration(
        sd_gen.generate_pipeline,
        gpu_duration_arg,
        *generation_args,
    )

    end_time = time.time()
    execution_time = end_time - start_time
    msg_task_complete = (
        f"GPU task complete in: {int(round(execution_time, 0) + 1)} seconds"
    )

    if verbose_arg:
        gr.Info(msg_task_complete)
        print(msg_task_complete)

    yield msg_task_complete, gr.update(), gr.update()


@spaces.GPU(duration=15)
def process_upscale(image, upscaler_name, upscaler_size):
    if image is None: return None

    from stablepy.diffusers_vanilla.utils import save_pil_image_with_metadata
    from stablepy import load_upscaler_model

    image = image.convert("RGB")
    exif_image = extract_exif_data(image)

    name_upscaler = UPSCALER_DICT_GUI[upscaler_name]

    if "https://" in str(name_upscaler):

        if not os.path.exists(f"./{DIRECTORY_UPSCALERS}/{name_upscaler.split('/')[-1]}"):
            download_things(DIRECTORY_UPSCALERS, name_upscaler, HF_TOKEN)

        name_upscaler = f"./{DIRECTORY_UPSCALERS}/{name_upscaler.split('/')[-1]}"

    scaler_beta = load_upscaler_model(model=name_upscaler, tile=0, tile_overlap=8, device="cuda", half=True)
    image_up = scaler_beta.upscale(image, upscaler_size, True)

    image_path = save_pil_image_with_metadata(image_up, f'{os.getcwd()}/up_images', exif_image)

    return image_path


# https://huggingface.co/spaces/BestWishYsh/ConsisID-preview-Space/discussions/1#674969a022b99c122af5d407
dynamic_gpu_duration.zerogpu = True
sd_gen_generate_pipeline.zerogpu = True
sd_gen = GuiSD()

with gr.Blocks(theme="NoCrypt/miku", css=CSS) as app:
    gr.Markdown("# 🧩 DiffuseCraft")
    gr.Markdown(SUBTITLE_GUI)
    with gr.Tab("Generation"):
        with gr.Row():

            with gr.Column(scale=2):

                def update_task_options(model_name, task_name):
                    new_choices = MODEL_TYPE_TASK[get_model_type(model_name)]

                    if task_name not in new_choices:
                        task_name = "txt2img"

                    return gr.update(value=task_name, choices=new_choices)

                task_gui = gr.Dropdown(label="Task", choices=SDXL_TASK, value=TASK_MODEL_LIST[0])
                model_name_gui = gr.Dropdown(label="Model", choices=model_list, value=model_list[0], allow_custom_value=True)
                prompt_gui = gr.Textbox(lines=5, placeholder="Enter prompt", label="Prompt")
                neg_prompt_gui = gr.Textbox(lines=3, placeholder="Enter Neg prompt", label="Negative prompt", value="lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, worst quality, low quality, very displeasing, (bad)")
                with gr.Row(equal_height=False):
                    set_params_gui = gr.Button(value="↙️", variant="secondary", size="sm")
                    clear_prompt_gui = gr.Button(value="🗑️", variant="secondary", size="sm")
                    set_random_seed = gr.Button(value="🎲", variant="secondary", size="sm")
                generate_button = gr.Button(value="GENERATE IMAGE", variant="primary")

                model_name_gui.change(
                    update_task_options,
                    [model_name_gui, task_gui],
                    [task_gui],
                )

                load_model_gui = gr.HTML(elem_id="load_model", elem_classes="contain")

                result_images = gr.Gallery(
                    label="Generated images",
                    show_label=False,
                    elem_id="gallery",
                    columns=[2],
                    rows=[2],
                    object_fit="contain",
                    # height="auto",
                    interactive=False,
                    preview=False,
                    selected_index=50,
                )

                actual_task_info = gr.HTML()

                with gr.Row(equal_height=False, variant="default"):
                    gpu_duration_gui = gr.Number(minimum=5, maximum=240, value=59, show_label=False, container=False, info="GPU time duration (seconds)")
                    with gr.Column():
                        verbose_info_gui = gr.Checkbox(value=False, container=False, label="Status info")
                        load_lora_cpu_gui = gr.Checkbox(value=False, container=False, label="Load LoRAs on CPU")

            with gr.Column(scale=1):
                steps_gui = gr.Slider(minimum=1, maximum=100, step=1, value=28, label="Steps")
                cfg_gui = gr.Slider(minimum=0, maximum=30, step=0.5, value=7., label="CFG")
                sampler_gui = gr.Dropdown(label="Sampler", choices=scheduler_names, value="Euler")
                schedule_type_gui = gr.Dropdown(label="Schedule type", choices=SCHEDULE_TYPE_OPTIONS, value=SCHEDULE_TYPE_OPTIONS[0])
                img_width_gui = gr.Slider(minimum=64, maximum=4096, step=8, value=1024, label="Img Width")
                img_height_gui = gr.Slider(minimum=64, maximum=4096, step=8, value=1024, label="Img Height")
                seed_gui = gr.Number(minimum=-1, maximum=9999999999, value=-1, label="Seed")
                pag_scale_gui = gr.Slider(minimum=0.0, maximum=10.0, step=0.1, value=0.0, label="PAG Scale")
                with gr.Row():
                    clip_skip_gui = gr.Checkbox(value=True, label="Layer 2 Clip Skip")
                    free_u_gui = gr.Checkbox(value=False, label="FreeU")

                with gr.Row(equal_height=False):

                    def run_set_params_gui(base_prompt, name_model):
                        valid_receptors = {  # default values
                            "prompt": gr.update(value=base_prompt),
                            "neg_prompt": gr.update(value=""),
                            "Steps": gr.update(value=30),
                            "width": gr.update(value=1024),
                            "height": gr.update(value=1024),
                            "Seed": gr.update(value=-1),
                            "Sampler": gr.update(value="Euler"),
                            "CFG scale": gr.update(value=7.),  # cfg
                            "Clip skip": gr.update(value=True),
                            "Model": gr.update(value=name_model),
                            "Schedule type": gr.update(value="Automatic"),
                            "PAG": gr.update(value=.0),
                            "FreeU": gr.update(value=False),
                        }
                        valid_keys = list(valid_receptors.keys())

                        parameters = extract_parameters(base_prompt)
                        # print(parameters)

                        if "Sampler" in parameters:
                            value_sampler = parameters["Sampler"]
                            for s_type in SCHEDULE_TYPE_OPTIONS:
                                if s_type in value_sampler:
                                    value_sampler = value_sampler.replace(s_type, "").strip()
                                    parameters["Sampler"] = value_sampler
                                    parameters["Schedule type"] = s_type

                        for key, val in parameters.items():
                            # print(val)
                            if key in valid_keys:
                                try:
                                    if key == "Sampler":
                                        if val not in scheduler_names:
                                            continue
                                    if key == "Schedule type":
                                        if val not in SCHEDULE_TYPE_OPTIONS:
                                            val = "Automatic"
                                    elif key == "Clip skip":
                                        if "," in str(val):
                                            val = val.replace(",", "")
                                        if int(val) >= 2:
                                            val = True
                                    if key == "prompt":
                                        if ">" in val and "<" in val:
                                            val = re.sub(r'<[^>]+>', '', val)
                                            print("Removed LoRA written in the prompt")
                                    if key in ["prompt", "neg_prompt"]:
                                        val = re.sub(r'\s+', ' ', re.sub(r',+', ',', val)).strip()
                                    if key in ["Steps", "width", "height", "Seed"]:
                                        val = int(val)
                                    if key == "FreeU":
                                        val = True
                                    if key in ["CFG scale", "PAG"]:
                                        val = float(val)
                                    if key == "Model":
                                        filtered_models = [m for m in model_list if val in m]
                                        if filtered_models:
                                            val = filtered_models[0]
                                        else:
                                            val = name_model
                                    if key == "Seed":
                                        continue
                                    valid_receptors[key] = gr.update(value=val)
                                    # print(val, type(val))
                                    # print(valid_receptors)
                                except Exception as e:
                                    print(str(e))
                        return [value for value in valid_receptors.values()]

                    set_params_gui.click(
                        run_set_params_gui, [prompt_gui, model_name_gui], [
                            prompt_gui,
                            neg_prompt_gui,
                            steps_gui,
                            img_width_gui,
                            img_height_gui,
                            seed_gui,
                            sampler_gui,
                            cfg_gui,
                            clip_skip_gui,
                            model_name_gui,
                            schedule_type_gui,
                            pag_scale_gui,
                            free_u_gui,
                        ],
                    )

                    def run_clear_prompt_gui():
                        return gr.update(value=""), gr.update(value="")
                    clear_prompt_gui.click(
                        run_clear_prompt_gui, [], [prompt_gui, neg_prompt_gui]
                    )

                    def run_set_random_seed():
                        return -1
                    set_random_seed.click(
                        run_set_random_seed, [], seed_gui
                    )

                num_images_gui = gr.Slider(minimum=1, maximum=5, step=1, value=1, label="Images")
                prompt_syntax_gui = gr.Dropdown(label="Prompt Syntax", choices=PROMPT_W_OPTIONS, value=PROMPT_W_OPTIONS[1][1])
                vae_model_gui = gr.Dropdown(label="VAE Model", choices=vae_model_list, value=vae_model_list[0])

                with gr.Accordion("Hires fix", open=False, visible=True):

                    upscaler_model_path_gui = gr.Dropdown(label="Upscaler", choices=UPSCALER_KEYS, value=UPSCALER_KEYS[0])
                    upscaler_increases_size_gui = gr.Slider(minimum=1.1, maximum=4., step=0.1, value=1.2, label="Upscale by")
                    upscaler_tile_size_gui = gr.Slider(minimum=0, maximum=512, step=16, value=0, label="Upscaler Tile Size", info="0 = no tiling")
                    upscaler_tile_overlap_gui = gr.Slider(minimum=0, maximum=48, step=1, value=8, label="Upscaler Tile Overlap")
                    hires_steps_gui = gr.Slider(minimum=0, value=30, maximum=100, step=1, label="Hires Steps")
                    hires_denoising_strength_gui = gr.Slider(minimum=0.1, maximum=1.0, step=0.01, value=0.55, label="Hires Denoising Strength")
                    hires_sampler_gui = gr.Dropdown(label="Hires Sampler", choices=POST_PROCESSING_SAMPLER, value=POST_PROCESSING_SAMPLER[0])
                    hires_schedule_list = ["Use same schedule type"] + SCHEDULE_TYPE_OPTIONS
                    hires_schedule_type_gui = gr.Dropdown(label="Hires Schedule type", choices=hires_schedule_list, value=hires_schedule_list[0])
                    hires_guidance_scale_gui = gr.Slider(minimum=-1., maximum=30., step=0.5, value=-1., label="Hires CFG", info="If the value is -1, the main CFG will be used")
                    hires_prompt_gui = gr.Textbox(label="Hires Prompt", placeholder="Main prompt will be use", lines=3)
                    hires_negative_prompt_gui = gr.Textbox(label="Hires Negative Prompt", placeholder="Main negative prompt will be use", lines=3)

                with gr.Accordion("LoRA", open=False, visible=True):

                    def lora_dropdown(label, visible=True):
                        return gr.Dropdown(label=label, choices=lora_model_list, value="None", allow_custom_value=True, visible=visible)

                    def lora_scale_slider(label, visible=True):
                        return gr.Slider(minimum=-2, maximum=2, step=0.01, value=0.33, label=label, visible=visible)

                    lora1_gui = lora_dropdown("Lora1")
                    lora_scale_1_gui = lora_scale_slider("Lora Scale 1")
                    lora2_gui = lora_dropdown("Lora2")
                    lora_scale_2_gui = lora_scale_slider("Lora Scale 2")
                    lora3_gui = lora_dropdown("Lora3")
                    lora_scale_3_gui = lora_scale_slider("Lora Scale 3")
                    lora4_gui = lora_dropdown("Lora4")
                    lora_scale_4_gui = lora_scale_slider("Lora Scale 4")
                    lora5_gui = lora_dropdown("Lora5")
                    lora_scale_5_gui = lora_scale_slider("Lora Scale 5")
                    lora6_gui = lora_dropdown("Lora6", visible=False)
                    lora_scale_6_gui = lora_scale_slider("Lora Scale 6", visible=False)
                    lora7_gui = lora_dropdown("Lora7", visible=False)
                    lora_scale_7_gui = lora_scale_slider("Lora Scale 7", visible=False)

                    with gr.Accordion("From URL", open=False, visible=True):
                        text_lora = gr.Textbox(
                            label="LoRA's download URL",
                            placeholder="https://civitai.com/api/download/models/28907",
                            lines=1,
                            info="It has to be .safetensors files, and you can also download them from Hugging Face.",
                        )
                        romanize_text = gr.Checkbox(value=False, label="Transliterate name", visible=False)
                        button_lora = gr.Button("Get and Refresh the LoRA Lists")
                        new_lora_status = gr.HTML()
                        button_lora.click(
                            get_my_lora,
                            [text_lora, romanize_text],
                            [lora1_gui, lora2_gui, lora3_gui, lora4_gui, lora5_gui, lora6_gui, lora7_gui, new_lora_status]
                        )

                with gr.Accordion("Face restoration", open=False, visible=True):

                    face_rest_options = [None] + FACE_RESTORATION_MODELS

                    face_restoration_model_gui = gr.Dropdown(label="Face restoration model", choices=face_rest_options, value=face_rest_options[0])
                    face_restoration_visibility_gui = gr.Slider(minimum=0., maximum=1., step=0.001, value=1., label="Visibility")
                    face_restoration_weight_gui = gr.Slider(minimum=0., maximum=1., step=0.001, value=.5, label="Weight", info="(0 = maximum effect, 1 = minimum effect)")

                with gr.Accordion("IP-Adapter", open=False, visible=True):

                    with gr.Accordion("IP-Adapter 1", open=False, visible=True):
                        image_ip1 = gr.Image(label="IP Image", type="filepath")
                        mask_ip1 = gr.Image(label="IP Mask", type="filepath")
                        model_ip1 = gr.Dropdown(value="plus_face", label="Model", choices=IP_MODELS)
                        mode_ip1 = gr.Dropdown(value="original", label="Mode", choices=MODE_IP_OPTIONS)
                        scale_ip1 = gr.Slider(minimum=0., maximum=2., step=0.01, value=0.7, label="Scale")
                    with gr.Accordion("IP-Adapter 2", open=False, visible=True):
                        image_ip2 = gr.Image(label="IP Image", type="filepath")
                        mask_ip2 = gr.Image(label="IP Mask (optional)", type="filepath")
                        model_ip2 = gr.Dropdown(value="base", label="Model", choices=IP_MODELS)
                        mode_ip2 = gr.Dropdown(value="style", label="Mode", choices=MODE_IP_OPTIONS)
                        scale_ip2 = gr.Slider(minimum=0., maximum=2., step=0.01, value=0.7, label="Scale")

                with gr.Accordion("ControlNet / Img2img / Inpaint", open=False, visible=True):
                    image_control = gr.Image(label="Image ControlNet/Inpaint/Img2img", type="filepath")
                    image_mask_gui = gr.Image(label="Image Mask", type="filepath")
                    strength_gui = gr.Slider(
                        minimum=0.01, maximum=1.0, step=0.01, value=0.55, label="Strength",
                        info="This option adjusts the level of changes for img2img, repaint and inpaint."
                    )
                    image_resolution_gui = gr.Slider(
                        minimum=64, maximum=2048, step=64, value=1024, label="Image Resolution",
                        info="The maximum proportional size of the generated image based on the uploaded image."
                    )
                    controlnet_model_gui = gr.Dropdown(label="ControlNet model", choices=DIFFUSERS_CONTROLNET_MODEL, value=DIFFUSERS_CONTROLNET_MODEL[0], allow_custom_value=True)
                    control_net_output_scaling_gui = gr.Slider(minimum=0, maximum=5.0, step=0.1, value=1, label="ControlNet Output Scaling in UNet")
                    control_net_start_threshold_gui = gr.Slider(minimum=0, maximum=1, step=0.01, value=0, label="ControlNet Start Threshold (%)")
                    control_net_stop_threshold_gui = gr.Slider(minimum=0, maximum=1, step=0.01, value=1, label="ControlNet Stop Threshold (%)")
                    preprocessor_name_gui = gr.Dropdown(label="Preprocessor Name", choices=TASK_AND_PREPROCESSORS["canny"])

                    def change_preprocessor_choices(task):
                        task = TASK_STABLEPY[task]
                        if task in TASK_AND_PREPROCESSORS.keys():
                            choices_task = TASK_AND_PREPROCESSORS[task]
                        else:
                            choices_task = TASK_AND_PREPROCESSORS["canny"]
                        return gr.update(choices=choices_task, value=choices_task[0])
                    task_gui.change(
                        change_preprocessor_choices,
                        [task_gui],
                        [preprocessor_name_gui],
                    )

                    preprocess_resolution_gui = gr.Slider(minimum=64, maximum=2048, step=64, value=512, label="Preprocessor Resolution")
                    low_threshold_gui = gr.Slider(minimum=1, maximum=255, step=1, value=100, label="'CANNY' low threshold")
                    high_threshold_gui = gr.Slider(minimum=1, maximum=255, step=1, value=200, label="'CANNY' high threshold")
                    value_threshold_gui = gr.Slider(minimum=1, maximum=2.0, step=0.01, value=0.1, label="'MLSD' Hough value threshold")
                    distance_threshold_gui = gr.Slider(minimum=1, maximum=20.0, step=0.01, value=0.1, label="'MLSD' Hough distance threshold")
                    recolor_gamma_correction_gui = gr.Number(minimum=0., maximum=25., value=1., step=0.001, label="'RECOLOR' gamma correction")
                    tile_blur_sigma_gui = gr.Number(minimum=0, maximum=100, value=9, step=1, label="'TILE' blur sigma")

                with gr.Accordion("T2I adapter", open=False, visible=False):
                    t2i_adapter_preprocessor_gui = gr.Checkbox(value=True, label="T2i Adapter Preprocessor")
                    adapter_conditioning_scale_gui = gr.Slider(minimum=0, maximum=5., step=0.1, value=1, label="Adapter Conditioning Scale")
                    adapter_conditioning_factor_gui = gr.Slider(minimum=0, maximum=1., step=0.01, value=0.55, label="Adapter Conditioning Factor (%)")

                with gr.Accordion("Styles", open=False, visible=True):

                    try:
                        style_names_found = sd_gen.model.STYLE_NAMES
                    except Exception:
                        style_names_found = STYLE_NAMES

                    style_prompt_gui = gr.Dropdown(
                        style_names_found,
                        multiselect=True,
                        value=None,
                        label="Style Prompt",
                        interactive=True,
                    )
                    style_json_gui = gr.File(label="Style JSON File")
                    style_button = gr.Button("Load styles")

                    def load_json_style_file(json):
                        if not sd_gen.model:
                            gr.Info("First load the model")
                            return gr.update(value=None, choices=STYLE_NAMES)

                        sd_gen.model.load_style_file(json)
                        gr.Info(f"{len(sd_gen.model.STYLE_NAMES)} styles loaded")
                        return gr.update(value=None, choices=sd_gen.model.STYLE_NAMES)

                    style_button.click(load_json_style_file, [style_json_gui], [style_prompt_gui])                        

                with gr.Accordion("Textual inversion", open=False, visible=False):
                    active_textual_inversion_gui = gr.Checkbox(value=False, label="Active Textual Inversion in prompt")

                with gr.Accordion("Detailfix", open=False, visible=True):

                    # Adetailer Inpaint Only
                    adetailer_inpaint_only_gui = gr.Checkbox(label="Inpaint only", value=True)

                    # Adetailer Verbose
                    adetailer_verbose_gui = gr.Checkbox(label="Verbose", value=False)

                    # Adetailer Sampler
                    adetailer_sampler_gui = gr.Dropdown(label="Adetailer sampler:", choices=POST_PROCESSING_SAMPLER, value=POST_PROCESSING_SAMPLER[0])

                    with gr.Accordion("Detailfix A", open=False, visible=True):
                        # Adetailer A
                        adetailer_active_a_gui = gr.Checkbox(label="Enable Adetailer A", value=False)
                        prompt_ad_a_gui = gr.Textbox(label="Main prompt", placeholder="Main prompt will be use", lines=3)
                        negative_prompt_ad_a_gui = gr.Textbox(label="Negative prompt", placeholder="Main negative prompt will be use", lines=3)
                        strength_ad_a_gui = gr.Number(label="Strength:", value=0.35, step=0.01, minimum=0.01, maximum=1.0)
                        face_detector_ad_a_gui = gr.Checkbox(label="Face detector", value=True)
                        person_detector_ad_a_gui = gr.Checkbox(label="Person detector", value=False)
                        hand_detector_ad_a_gui = gr.Checkbox(label="Hand detector", value=False)
                        mask_dilation_a_gui = gr.Number(label="Mask dilation:", value=4, minimum=1)
                        mask_blur_a_gui = gr.Number(label="Mask blur:", value=4, minimum=1)
                        mask_padding_a_gui = gr.Number(label="Mask padding:", value=32, minimum=1)

                    with gr.Accordion("Detailfix B", open=False, visible=True):
                        # Adetailer B
                        adetailer_active_b_gui = gr.Checkbox(label="Enable Adetailer B", value=False)
                        prompt_ad_b_gui = gr.Textbox(label="Main prompt", placeholder="Main prompt will be use", lines=3)
                        negative_prompt_ad_b_gui = gr.Textbox(label="Negative prompt", placeholder="Main negative prompt will be use", lines=3)
                        strength_ad_b_gui = gr.Number(label="Strength:", value=0.35, step=0.01, minimum=0.01, maximum=1.0)
                        face_detector_ad_b_gui = gr.Checkbox(label="Face detector", value=False)
                        person_detector_ad_b_gui = gr.Checkbox(label="Person detector", value=True)
                        hand_detector_ad_b_gui = gr.Checkbox(label="Hand detector", value=False)
                        mask_dilation_b_gui = gr.Number(label="Mask dilation:", value=4, minimum=1)
                        mask_blur_b_gui = gr.Number(label="Mask blur:", value=4, minimum=1)
                        mask_padding_b_gui = gr.Number(label="Mask padding:", value=32, minimum=1)

                with gr.Accordion("Other settings", open=False, visible=True):
                    schedule_prediction_type_gui = gr.Dropdown(label="Discrete Sampling Type", choices=SCHEDULE_PREDICTION_TYPE_OPTIONS, value=SCHEDULE_PREDICTION_TYPE_OPTIONS[0])
                    guidance_rescale_gui = gr.Number(label="CFG rescale:", value=0., step=0.01, minimum=0., maximum=1.5)
                    save_generated_images_gui = gr.Checkbox(value=True, label="Create a download link for the images")
                    filename_pattern_gui = gr.Textbox(label="Filename pattern", value="model,seed", placeholder="model,seed,sampler,schedule_type,img_width,img_height,guidance_scale,num_steps,vae,prompt_section,neg_prompt_section", lines=1)
                    hires_before_adetailer_gui = gr.Checkbox(value=False, label="Hires Before Adetailer")
                    hires_after_adetailer_gui = gr.Checkbox(value=True, label="Hires After Adetailer")
                    generator_in_cpu_gui = gr.Checkbox(value=False, label="Generator in CPU")

                with gr.Accordion("More settings", open=False, visible=False):
                    loop_generation_gui = gr.Slider(minimum=1, value=1, label="Loop Generation")
                    retain_task_cache_gui = gr.Checkbox(value=False, label="Retain task model in cache")
                    leave_progress_bar_gui = gr.Checkbox(value=True, label="Leave Progress Bar")
                    disable_progress_bar_gui = gr.Checkbox(value=False, label="Disable Progress Bar")
                    display_images_gui = gr.Checkbox(value=False, label="Display Images")
                    image_previews_gui = gr.Checkbox(value=True, label="Image Previews")
                    image_storage_location_gui = gr.Textbox(value="./images", label="Image Storage Location")
                    retain_compel_previous_load_gui = gr.Checkbox(value=False, label="Retain Compel Previous Load")
                    retain_detailfix_model_previous_load_gui = gr.Checkbox(value=False, label="Retain Detailfix Model Previous Load")
                    retain_hires_model_previous_load_gui = gr.Checkbox(value=False, label="Retain Hires Model Previous Load")
                    xformers_memory_efficient_attention_gui = gr.Checkbox(value=False, label="Xformers Memory Efficient Attention")

        with gr.Accordion("Examples and help", open=False, visible=True):
            gr.Markdown(HELP_GUI)
            gr.Markdown(EXAMPLES_GUI_HELP)
            gr.Examples(
                examples=EXAMPLES_GUI,
                fn=sd_gen.generate_pipeline,
                inputs=[
                    prompt_gui,
                    neg_prompt_gui,
                    steps_gui,
                    cfg_gui,
                    seed_gui,
                    lora1_gui,
                    lora_scale_1_gui,
                    sampler_gui,
                    img_height_gui,
                    img_width_gui,
                    model_name_gui,
                    task_gui,
                    image_control,
                    image_resolution_gui,
                    strength_gui,
                    control_net_output_scaling_gui,
                    control_net_start_threshold_gui,
                    control_net_stop_threshold_gui,
                    prompt_syntax_gui,
                    upscaler_model_path_gui,
                    gpu_duration_gui,
                    load_lora_cpu_gui,
                ],
                outputs=[load_model_gui, result_images, actual_task_info],
                cache_examples=False,
            )
            gr.Markdown(RESOURCES)

    with gr.Tab("Inpaint mask maker", render=True):

        with gr.Row():
            with gr.Column(scale=2):
                image_base = gr.ImageEditor(
                    sources=["upload", "clipboard"],
                    # crop_size="1:1",
                    # enable crop (or disable it)
                    # transforms=["crop"],
                    brush=gr.Brush(
                        default_size="16",  # or leave it as 'auto'
                        color_mode="fixed",  # 'fixed' hides the user swatches and colorpicker, 'defaults' shows it
                        # default_color="black", # html names are supported
                        colors=[
                            "rgba(0, 0, 0, 1)",  # rgb(a)
                            "rgba(0, 0, 0, 0.1)",
                            "rgba(255, 255, 255, 0.1)",
                            # "hsl(360, 120, 120)" # in fact any valid colorstring
                        ]
                    ),
                    eraser=gr.Eraser(default_size="16")
                )
                invert_mask = gr.Checkbox(value=False, label="Invert mask")
                btn = gr.Button("Create mask")
            with gr.Column(scale=1):
                img_source = gr.Image(interactive=False)
                img_result = gr.Image(label="Mask image", show_label=True, interactive=False)
                btn_send = gr.Button("Send to the first tab")

            btn.click(create_mask_now, [image_base, invert_mask], [img_source, img_result])

            def send_img(img_source, img_result):
                return img_source, img_result
            btn_send.click(send_img, [img_source, img_result], [image_control, image_mask_gui])

    with gr.Tab("PNG Info"):

        with gr.Row():
            with gr.Column():
                image_metadata = gr.Image(label="Image with metadata", type="pil", sources=["upload"])

            with gr.Column():
                result_metadata = gr.Textbox(label="Metadata", show_label=True, show_copy_button=True, interactive=False, container=True, max_lines=99)

                image_metadata.change(
                    fn=extract_exif_data,
                    inputs=[image_metadata],
                    outputs=[result_metadata],
                )

    with gr.Tab("Upscaler"):

        with gr.Row():
            with gr.Column():

                USCALER_TAB_KEYS = [name for name in UPSCALER_KEYS[9:]]

                image_up_tab = gr.Image(label="Image", type="pil", sources=["upload"])
                upscaler_tab = gr.Dropdown(label="Upscaler", choices=USCALER_TAB_KEYS, value=USCALER_TAB_KEYS[5])
                upscaler_size_tab = gr.Slider(minimum=1., maximum=4., step=0.1, value=1.1, label="Upscale by")
                generate_button_up_tab = gr.Button(value="START UPSCALE", variant="primary")

            with gr.Column():
                result_up_tab = gr.Image(label="Result", type="pil", interactive=False, format="png")

                generate_button_up_tab.click(
                    fn=process_upscale,
                    inputs=[image_up_tab, upscaler_tab, upscaler_size_tab],
                    outputs=[result_up_tab],
                )

    with gr.Tab("Preprocessor", render=True):
        preprocessor_tab()

    generate_button.click(
        fn=sd_gen.load_new_model,
        inputs=[
            model_name_gui,
            vae_model_gui,
            task_gui,
            controlnet_model_gui,
        ],
        outputs=[load_model_gui],
        queue=True,
        show_progress="minimal",
    ).success(
        fn=sd_gen_generate_pipeline,  # fn=sd_gen.generate_pipeline,
        inputs=[
            prompt_gui,
            neg_prompt_gui,
            num_images_gui,
            steps_gui,
            cfg_gui,
            clip_skip_gui,
            seed_gui,
            lora1_gui,
            lora_scale_1_gui,
            lora2_gui,
            lora_scale_2_gui,
            lora3_gui,
            lora_scale_3_gui,
            lora4_gui,
            lora_scale_4_gui,
            lora5_gui,
            lora_scale_5_gui,
            lora6_gui,
            lora_scale_6_gui,
            lora7_gui,
            lora_scale_7_gui,
            sampler_gui,
            schedule_type_gui,
            schedule_prediction_type_gui,
            img_height_gui,
            img_width_gui,
            model_name_gui,
            vae_model_gui,
            task_gui,
            image_control,
            preprocessor_name_gui,
            preprocess_resolution_gui,
            image_resolution_gui,
            style_prompt_gui,
            style_json_gui,
            image_mask_gui,
            strength_gui,
            low_threshold_gui,
            high_threshold_gui,
            value_threshold_gui,
            distance_threshold_gui,
            recolor_gamma_correction_gui,
            tile_blur_sigma_gui,
            control_net_output_scaling_gui,
            control_net_start_threshold_gui,
            control_net_stop_threshold_gui,
            active_textual_inversion_gui,
            prompt_syntax_gui,
            upscaler_model_path_gui,
            upscaler_increases_size_gui,
            upscaler_tile_size_gui,
            upscaler_tile_overlap_gui,
            hires_steps_gui,
            hires_denoising_strength_gui,
            hires_sampler_gui,
            hires_prompt_gui,
            hires_negative_prompt_gui,
            hires_before_adetailer_gui,
            hires_after_adetailer_gui,
            hires_schedule_type_gui,
            hires_guidance_scale_gui,
            controlnet_model_gui,
            loop_generation_gui,
            leave_progress_bar_gui,
            disable_progress_bar_gui,
            image_previews_gui,
            display_images_gui,
            save_generated_images_gui,
            filename_pattern_gui,
            image_storage_location_gui,
            retain_compel_previous_load_gui,
            retain_detailfix_model_previous_load_gui,
            retain_hires_model_previous_load_gui,
            t2i_adapter_preprocessor_gui,
            adapter_conditioning_scale_gui,
            adapter_conditioning_factor_gui,
            xformers_memory_efficient_attention_gui,
            free_u_gui,
            generator_in_cpu_gui,
            adetailer_inpaint_only_gui,
            adetailer_verbose_gui,
            adetailer_sampler_gui,
            adetailer_active_a_gui,
            prompt_ad_a_gui,
            negative_prompt_ad_a_gui,
            strength_ad_a_gui,
            face_detector_ad_a_gui,
            person_detector_ad_a_gui,
            hand_detector_ad_a_gui,
            mask_dilation_a_gui,
            mask_blur_a_gui,
            mask_padding_a_gui,
            adetailer_active_b_gui,
            prompt_ad_b_gui,
            negative_prompt_ad_b_gui,
            strength_ad_b_gui,
            face_detector_ad_b_gui,
            person_detector_ad_b_gui,
            hand_detector_ad_b_gui,
            mask_dilation_b_gui,
            mask_blur_b_gui,
            mask_padding_b_gui,
            retain_task_cache_gui,
            guidance_rescale_gui,
            image_ip1,
            mask_ip1,
            model_ip1,
            mode_ip1,
            scale_ip1,
            image_ip2,
            mask_ip2,
            model_ip2,
            mode_ip2,
            scale_ip2,
            pag_scale_gui,
            face_restoration_model_gui,
            face_restoration_visibility_gui,
            face_restoration_weight_gui,
            load_lora_cpu_gui,
            verbose_info_gui,
            gpu_duration_gui,
        ],
        outputs=[load_model_gui, result_images, actual_task_info],
        queue=True,
        show_progress="minimal",
    )

app.queue()

app.launch(
    show_error=True,
    debug=True,
    allowed_paths=["./images/"],
)