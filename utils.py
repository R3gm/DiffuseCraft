import os
import re
import gradio as gr
from constants import (
    DIFFUSERS_FORMAT_LORAS,
    CIVITAI_API_KEY,
    HF_TOKEN,
    MODEL_TYPE_CLASS,
    DIRECTORY_LORAS,
    DIRECTORY_MODELS,
    DIFFUSECRAFT_CHECKPOINT_NAME,
    CACHE_HF,
    STORAGE_ROOT,
)
from huggingface_hub import HfApi
from huggingface_hub import snapshot_download
from diffusers import DiffusionPipeline
from huggingface_hub import model_info as model_info_data
from diffusers.pipelines.pipeline_loading_utils import variant_compatible_siblings
from stablepy.diffusers_vanilla.utils import checkpoint_model_type
from pathlib import PosixPath
from unidecode import unidecode
import urllib.parse
import copy
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import shutil
import subprocess

USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0'


def request_json_data(url):
    model_version_id = url.split('/')[-1]
    if "?modelVersionId=" in model_version_id:
        match = re.search(r'modelVersionId=(\d+)', url)
        model_version_id = match.group(1)

    endpoint_url = f"https://civitai.com/api/v1/model-versions/{model_version_id}"

    params = {}
    headers = {'User-Agent': USER_AGENT, 'content-type': 'application/json'}
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))

    try:
        result = session.get(endpoint_url, params=params, headers=headers, stream=True, timeout=(3.0, 15))
        result.raise_for_status()
        json_data = result.json()
        return json_data if json_data else None
    except Exception as e:
        print(f"Error: {e}")
        return None


class ModelInformation:
    def __init__(self, json_data):
        self.model_version_id = json_data.get("id", "")
        self.model_id = json_data.get("modelId", "")
        self.download_url = json_data.get("downloadUrl", "")
        self.model_url = f"https://civitai.com/models/{self.model_id}?modelVersionId={self.model_version_id}"
        self.filename_url = next(
            (v.get("name", "") for v in reversed(json_data.get("files", [])) if str(self.model_version_id) in v.get("downloadUrl", "")), ""
        )
        self.filename_url = self.filename_url if self.filename_url else ""
        self.description = json_data.get("description", "")
        if self.description is None: self.description = ""
        self.model_name = json_data.get("model", {}).get("name", "")
        self.model_type = json_data.get("model", {}).get("type", "")
        self.nsfw = json_data.get("model", {}).get("nsfw", False)
        self.poi = json_data.get("model", {}).get("poi", False)
        self.images = [img.get("url", "") for img in json_data.get("images", [])]
        self.example_prompt = json_data.get("trainedWords", [""])[0] if json_data.get("trainedWords") else ""
        self.original_json = copy.deepcopy(json_data)


def retrieve_model_info(url):
    json_data = request_json_data(url)
    if not json_data:
        return None
    model_descriptor = ModelInformation(json_data)
    return model_descriptor


def download_things(directory, url, hf_token="", civitai_api_key="", romanize=False):
    url = url.strip()
    downloaded_file_path = None

    if "drive.google.com" in url:
        original_dir = os.getcwd()
        os.chdir(directory)
        os.system(f"gdown --fuzzy {url}")
        os.chdir(original_dir)
    elif "huggingface.co" in url:
        url = url.replace("?download=true", "")
        # url = urllib.parse.quote(url, safe=':/')  # fix encoding
        if "/blob/" in url:
            url = url.replace("/blob/", "/resolve/")
        user_header = f'"Authorization: Bearer {hf_token}"'

        filename = unidecode(url.split('/')[-1]) if romanize else url.split('/')[-1]

        if hf_token:
            os.system(f"aria2c --console-log-level=error --summary-interval=10 --header={user_header} -c -x 16 -k 1M -s 16 {url} -d {directory}  -o {filename}")
        else:
            os.system(f"aria2c --optimize-concurrent-downloads --console-log-level=error --summary-interval=10 -c -x 16 -k 1M -s 16 {url} -d {directory}  -o {filename}")

        downloaded_file_path = os.path.join(directory, filename)

    elif "civitai.com" in url:

        if not civitai_api_key:
            print("\033[91mYou need an API key to download Civitai models.\033[0m")

        model_profile = retrieve_model_info(url)
        if (
            model_profile is not None
            and model_profile.download_url
            and model_profile.filename_url
        ):
            url = model_profile.download_url
            filename = unidecode(model_profile.filename_url) if romanize else model_profile.filename_url
        else:
            if "?" in url:
                url = url.split("?")[0]
            filename = ""

        url_dl = url + f"?token={civitai_api_key}"
        print(f"Filename: {filename}")

        param_filename = ""
        if filename:
            param_filename = f"-o '{filename}'"

        aria2_command = (
            f'aria2c --console-log-level=error --summary-interval=10 -c -x 16 '
            f'-k 1M -s 16 -d "{directory}" {param_filename} "{url_dl}"'
        )
        os.system(aria2_command)

        if param_filename and os.path.exists(os.path.join(directory, filename)):
            downloaded_file_path = os.path.join(directory, filename)

        # # PLAN B
        # # Follow the redirect to get the actual download URL
        # curl_command = (
        #     f'curl -L -sI --connect-timeout 5 --max-time 5 '
        #     f'-H "Content-Type: application/json" '
        #     f'-H "Authorization: Bearer {civitai_api_key}" "{url}"'
        # )

        # headers = os.popen(curl_command).read()

        # # Look for the redirected "Location" URL
        # location_match = re.search(r'location: (.+)', headers, re.IGNORECASE)

        # if location_match:
        #     redirect_url = location_match.group(1).strip()

        #     # Extract the filename from the redirect URL's "Content-Disposition"
        #     filename_match = re.search(r'filename%3D%22(.+?)%22', redirect_url)
        #     if filename_match:
        #         encoded_filename = filename_match.group(1)
        #         # Decode the URL-encoded filename
        #         decoded_filename = urllib.parse.unquote(encoded_filename)

        #         filename = unidecode(decoded_filename) if romanize else decoded_filename
        #         print(f"Filename: {filename}")

        #         aria2_command = (
        #             f'aria2c --console-log-level=error --summary-interval=10 -c -x 16 '
        #             f'-k 1M -s 16 -d "{directory}" -o "{filename}" "{redirect_url}"'
        #         )
        #         return_code = os.system(aria2_command)

        #         # if return_code != 0:
        #         #     raise RuntimeError(f"Failed to download file: {filename}. Error code: {return_code}")
        #         downloaded_file_path = os.path.join(directory, filename)
        #         if not os.path.exists(downloaded_file_path):
        #             downloaded_file_path = None

        # if not downloaded_file_path:
        #     # Old method
        #     if "?" in url:
        #         url = url.split("?")[0]
        #     url = url + f"?token={civitai_api_key}"
        #     os.system(f"aria2c --console-log-level=error --summary-interval=10 -c -x 16 -k 1M -s 16 -d {directory} {url}")

    else:
        os.system(f"aria2c --console-log-level=error --summary-interval=10 -c -x 16 -k 1M -s 16 -d {directory} {url}")

    return downloaded_file_path


def get_model_list(directory_path):
    model_list = []
    valid_extensions = {'.ckpt', '.pt', '.pth', '.safetensors', '.bin'}

    for filename in os.listdir(directory_path):
        if os.path.splitext(filename)[1] in valid_extensions:
            # name_without_extension = os.path.splitext(filename)[0]
            file_path = os.path.join(directory_path, filename)
            # model_list.append((name_without_extension, file_path))
            model_list.append(file_path)
            print('\033[34mFILE: ' + file_path + '\033[0m')
    return model_list


def extract_parameters(input_string):
    parameters = {}
    input_string = input_string.replace("\n", "")

    if "Negative prompt:" not in input_string:
        if "Steps:" in input_string:
            input_string = input_string.replace("Steps:", "Negative prompt: Steps:")
        else:
            print("Invalid metadata")
            parameters["prompt"] = input_string
            return parameters

    parm = input_string.split("Negative prompt:")
    parameters["prompt"] = parm[0].strip()
    if "Steps:" not in parm[1]:
        print("Steps not detected")
        parameters["neg_prompt"] = parm[1].strip()
        return parameters
    parm = parm[1].split("Steps:")
    parameters["neg_prompt"] = parm[0].strip()
    input_string = "Steps:" + parm[1]

    # Extracting Steps
    steps_match = re.search(r'Steps: (\d+)', input_string)
    if steps_match:
        parameters['Steps'] = int(steps_match.group(1))

    # Extracting Size
    size_match = re.search(r'Size: (\d+x\d+)', input_string)
    if size_match:
        parameters['Size'] = size_match.group(1)
        width, height = map(int, parameters['Size'].split('x'))
        parameters['width'] = width
        parameters['height'] = height

    # Extracting other parameters
    other_parameters = re.findall(r'([^,:]+): (.*?)(?=, [^,:]+:|$)', input_string)
    for param in other_parameters:
        parameters[param[0].strip()] = param[1].strip('"')

    return parameters


def get_my_lora(link_url, romanize):
    l_name = ""
    for url in [url.strip() for url in link_url.split(',')]:
        if not os.path.exists(f"./loras/{url.split('/')[-1]}"):
            l_name = download_things(DIRECTORY_LORAS, url, HF_TOKEN, CIVITAI_API_KEY, romanize)
    new_lora_model_list = get_model_list(DIRECTORY_LORAS)
    new_lora_model_list.insert(0, "None")
    new_lora_model_list = new_lora_model_list + DIFFUSERS_FORMAT_LORAS
    msg_lora = "Downloaded"
    if l_name:
        msg_lora += f": <b>{l_name}</b>"
        print(msg_lora)

    return gr.update(
        choices=new_lora_model_list
    ), gr.update(
        choices=new_lora_model_list
    ), gr.update(
        choices=new_lora_model_list
    ), gr.update(
        choices=new_lora_model_list
    ), gr.update(
        choices=new_lora_model_list
    ), gr.update(
        choices=new_lora_model_list
    ), gr.update(
        choices=new_lora_model_list
    ), gr.update(
        value=msg_lora
    )


def info_html(json_data, title, subtitle):
    return f"""
        <div style='padding: 0; border-radius: 10px;'>
            <p style='margin: 0; font-weight: bold;'>{title}</p>
            <details>
                <summary>Details</summary>
                <p style='margin: 0; font-weight: bold;'>{subtitle}</p>
            </details>
        </div>
        """


def get_model_type(repo_id: str):
    api = HfApi(token=os.environ.get("HF_TOKEN"))  # if use private or gated model
    default = "SD 1.5"
    try:
        if os.path.exists(repo_id):
            tag = checkpoint_model_type(repo_id)
            return DIFFUSECRAFT_CHECKPOINT_NAME[tag]
        else:
            model = api.model_info(repo_id=repo_id, timeout=5.0)
            tags = model.tags
            for tag in tags:
                if tag in MODEL_TYPE_CLASS.keys(): return MODEL_TYPE_CLASS.get(tag, default)

    except Exception:
        return default
    return default


def restart_space(repo_id: str, factory_reboot: bool):
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    try:
        runtime = api.get_space_runtime(repo_id=repo_id)
        if runtime.stage == "RUNNING":
            api.restart_space(repo_id=repo_id, factory_reboot=factory_reboot)
            print(f"Restarting space: {repo_id}")
        else:
            print(f"Space {repo_id} is in stage: {runtime.stage}")
    except Exception as e:
        print(e)


def extract_exif_data(image):
    if image is None:
        return ""

    try:
        metadata_keys = ['parameters', 'metadata', 'prompt', 'Comment']

        for key in metadata_keys:
            if key in image.info:
                return image.info[key]

        return str(image.info)

    except Exception as e:
        return f"Error extracting metadata: {str(e)}"


def create_mask_now(img, invert):
    import numpy as np
    import time

    time.sleep(0.5)

    transparent_image = img["layers"][0]

    # Extract the alpha channel
    alpha_channel = np.array(transparent_image)[:, :, 3]

    # Create a binary mask by thresholding the alpha channel
    binary_mask = alpha_channel > 1

    if invert:
        print("Invert")
        # Invert the binary mask so that the drawn shape is white and the rest is black
        binary_mask = np.invert(binary_mask)

    # Convert the binary mask to a 3-channel RGB mask
    rgb_mask = np.stack((binary_mask,) * 3, axis=-1)

    # Convert the mask to uint8
    rgb_mask = rgb_mask.astype(np.uint8) * 255

    return img["background"], rgb_mask


def download_diffuser_repo(repo_name: str, model_type: str, revision: str = "main", token=True):

    variant = None
    if token is True and not os.environ.get("HF_TOKEN"):
        token = None

    if model_type == "SDXL":
        info = model_info_data(
            repo_name,
            token=token,
            revision=revision,
            timeout=5.0,
        )

        filenames = {sibling.rfilename for sibling in info.siblings}
        model_filenames, variant_filenames = variant_compatible_siblings(
            filenames, variant="fp16"
        )

        if len(variant_filenames):
            variant = "fp16"

    if model_type == "FLUX":
        cached_folder = snapshot_download(
            repo_id=repo_name,
            allow_patterns="transformer/*"
        )
    else:
        cached_folder = DiffusionPipeline.download(
            pretrained_model_name=repo_name,
            force_download=False,
            token=token,
            revision=revision,
            # mirror="https://hf-mirror.com",
            variant=variant,
            use_safetensors=True,
            trust_remote_code=False,
            timeout=5.0,
        )

    if isinstance(cached_folder, PosixPath):
        cached_folder = cached_folder.as_posix()

    # Task model
    # from huggingface_hub import hf_hub_download
    # hf_hub_download(
    #     task_model,
    #     filename="diffusion_pytorch_model.safetensors",  # fix fp16 variant
    # )

    return cached_folder


def get_folder_size_gb(folder_path):
    result = subprocess.run(["du", "-s", folder_path], capture_output=True, text=True)

    total_size_kb = int(result.stdout.split()[0])
    total_size_gb = total_size_kb / (1024 ** 2)

    return total_size_gb


def get_used_storage_gb():
    try:
        used_gb = get_folder_size_gb(STORAGE_ROOT)
        print(f"Used Storage: {used_gb:.2f} GB")
    except Exception as e:
        used_gb = 999
        print(f"Error while retrieving the used storage: {e}.")

    return used_gb


def delete_model(removal_candidate):
    print(f"Removing: {removal_candidate}")

    if os.path.exists(removal_candidate):
        os.remove(removal_candidate)
    else:
        diffusers_model = f"{CACHE_HF}{DIRECTORY_MODELS}--{removal_candidate.replace('/', '--')}"
        if os.path.isdir(diffusers_model):
            shutil.rmtree(diffusers_model)


def progress_step_bar(step, total):
    # Calculate the percentage for the progress bar width
    percentage = min(100, ((step / total) * 100))

    return f"""
        <div style="position: relative; width: 100%; background-color: gray; border-radius: 5px; overflow: hidden;">
            <div style="width: {percentage}%; height: 17px; background-color: #800080; transition: width 0.5s;"></div>
            <div style="position: absolute; width: 100%; text-align: center; color: white; top: 0; line-height: 19px; font-size: 13px;">
                {int(percentage)}%
            </div>
        </div>
        """


def html_template_message(msg):
    return f"""
        <div style="position: relative; width: 100%; background-color: gray; border-radius: 5px; overflow: hidden;">
            <div style="width: 0%; height: 17px; background-color: #800080; transition: width 0.5s;"></div>
            <div style="position: absolute; width: 100%; text-align: center; color: white; top: 0; line-height: 19px; font-size: 14px; font-weight: bold; text-shadow: 1px 1px 2px black;">
                {msg}
            </div>
        </div>
        """


def escape_html(text):
    """Escapes HTML special characters in the input text."""
    return text.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
