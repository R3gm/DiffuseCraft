import os
from stablepy.diffusers_vanilla.constants import FLUX_CN_UNION_MODES
from stablepy import (
    scheduler_names,
    SD15_TASKS,
    SDXL_TASKS,
    ALL_BUILTIN_UPSCALERS,
    IP_ADAPTERS_SD,
    IP_ADAPTERS_SDXL,
)

# - **Download Models**
DOWNLOAD_MODEL = "https://huggingface.co/TechnoByte/MilkyWonderland/resolve/main/milkyWonderland_v40.safetensors"

# - **Download VAEs**
DOWNLOAD_VAE = "https://huggingface.co/fp16-guy/anything_kl-f8-anime2_vae-ft-mse-840000-ema-pruned_blessed_clearvae_fp16_cleaned/resolve/main/vae-ft-mse-840000-ema-pruned_fp16.safetensors?download=true"

# - **Download LoRAs**
DOWNLOAD_LORA = "https://huggingface.co/Leopain/color/resolve/main/Coloring_book_-_LineArt.safetensors, https://civitai.com/api/download/models/135867, https://huggingface.co/Linaqruf/anime-detailer-xl-lora/resolve/main/anime-detailer-xl.safetensors?download=true, https://huggingface.co/Linaqruf/style-enhancer-xl-lora/resolve/main/style-enhancer-xl.safetensors?download=true, https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-SD15-8steps-CFG-lora.safetensors?download=true, https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-SDXL-8steps-CFG-lora.safetensors?download=true"

LOAD_DIFFUSERS_FORMAT_MODEL = [
    'stabilityai/stable-diffusion-xl-base-1.0',
    'Laxhar/noobai-XL-1.1',
    'Laxhar/noobai-XL-Vpred-1.0',
    'black-forest-labs/FLUX.1-dev',
    'John6666/blue-pencil-flux1-v021-fp8-flux',
    'John6666/wai-ani-flux-v10forfp8-fp8-flux',
    'John6666/xe-anime-flux-v04-fp8-flux',
    'John6666/lyh-anime-flux-v2a1-fp8-flux',
    'John6666/carnival-unchained-v10-fp8-flux',
    'John6666/iniverse-mix-xl-sfwnsfw-fluxdfp16nsfwv11-fp8-flux',
    'Freepik/flux.1-lite-8B-alpha',
    'shauray/FluxDev-HyperSD-merged',
    'mikeyandfriends/PixelWave_FLUX.1-dev_03',
    'terminusresearch/FluxBooru-v0.3',
    'black-forest-labs/FLUX.1-schnell',
    # 'ostris/OpenFLUX.1',
    'shuttleai/shuttle-3-diffusion',
    'Laxhar/noobai-XL-1.0',
    'John6666/noobai-xl-nai-xl-epsilonpred10version-sdxl',
    'Laxhar/noobai-XL-0.77',
    'John6666/noobai-xl-nai-xl-epsilonpred075version-sdxl',
    'Laxhar/noobai-XL-0.6',
    'John6666/noobai-xl-nai-xl-epsilonpred05version-sdxl',
    'John6666/noobai-cyberfix-v10-sdxl',
    'John6666/noobaiiter-xl-vpred-v075-sdxl',
    'John6666/ntr-mix-illustrious-xl-noob-xl-v40-sdxl',
    'John6666/ntr-mix-illustrious-xl-noob-xl-ntrmix35-sdxl',
    'John6666/ntr-mix-illustrious-xl-noob-xl-v777-sdxl',
    'John6666/ntr-mix-illustrious-xl-noob-xl-v777forlora-sdxl',
    'John6666/ntr-mix-illustrious-xl-noob-xl-xi-sdxl',
    'John6666/ntr-mix-illustrious-xl-noob-xl-xii-sdxl',
    'John6666/ntr-mix-illustrious-xl-noob-xl-xiii-sdxl',
    'John6666/mistoon-anime-v10illustrious-sdxl',
    'John6666/hassaku-xl-illustrious-v10-sdxl',
    'John6666/hassaku-xl-illustrious-v10style-sdxl',
    'John6666/haruki-mix-illustrious-v10-sdxl',
    'John6666/noobreal-v10-sdxl',
    'John6666/complicated-noobai-merge-vprediction-sdxl',
    'Laxhar/noobai-XL-Vpred-0.9r',
    'Laxhar/noobai-XL-Vpred-0.75s',
    'Laxhar/noobai-XL-Vpred-0.75',
    'Laxhar/noobai-XL-Vpred-0.65s',
    'Laxhar/noobai-XL-Vpred-0.65',
    'Laxhar/noobai-XL-Vpred-0.6',
    'John6666/cat-tower-noobai-xl-checkpoint-v14vpred-sdxl',
    'John6666/noobai-xl-nai-xl-vpred05version-sdxl',
    'John6666/noobai-fusion2-vpred-itercomp-v1-sdxl',
    'John6666/noobai-xl-nai-xl-vpredtestversion-sdxl',
    'John6666/chadmix-noobai075-illustrious01-v10-sdxl',
    'OnomaAIResearch/Illustrious-xl-early-release-v0',
    'John6666/illustriousxl-mmmix-v50-sdxl',
    'John6666/illustrious-pencil-xl-v200-sdxl',
    'John6666/obsession-illustriousxl-v21-sdxl',
    'John6666/obsession-illustriousxl-v30-sdxl',
    'John6666/obsession-illustriousxl-v31-sdxl',
    'John6666/wai-nsfw-illustrious-v70-sdxl',
    'John6666/illustrious-pony-mix-v3-sdxl',
    'John6666/nova-anime-xl-illustriousv10-sdxl',
    'John6666/nova-orange-xl-v30-sdxl',
    'John6666/silvermoon-mix03-illustrious-v10-sdxl',
    'eienmojiki/Anything-XL',
    'eienmojiki/Starry-XL-v5.2',
    'John6666/meinaxl-v2-sdxl',
    'Eugeoter/artiwaifu-diffusion-2.0',
    'comin/IterComp',
    'John6666/epicrealism-xl-vxiabeast-sdxl',
    'John6666/epicrealism-xl-v10kiss2-sdxl',
    'John6666/epicrealism-xl-v8kiss-sdxl',
    'misri/zavychromaxl_v80',
    'SG161222/RealVisXL_V4.0',
    'SG161222/RealVisXL_V5.0',
    'misri/newrealityxlAllInOne_Newreality40',
    'gsdf/CounterfeitXL',
    'WhiteAiZ/autismmixSDXL_autismmixConfetti_diffusers',
    'kitty7779/ponyDiffusionV6XL',
    'GraydientPlatformAPI/aniverse-pony',
    'John6666/ras-real-anime-screencap-v1-sdxl',
    'John6666/duchaiten-pony-xl-no-score-v60-sdxl',
    'John6666/mistoon-anime-ponyalpha-sdxl',
    'John6666/mistoon-xl-copper-v20fast-sdxl',
    'John6666/ebara-mfcg-pony-mix-v12-sdxl',
    'John6666/t-ponynai3-v51-sdxl',
    'John6666/t-ponynai3-v65-sdxl',
    'John6666/prefect-pony-xl-v3-sdxl',
    'John6666/prefect-pony-xl-v4-sdxl',
    'John6666/mala-anime-mix-nsfw-pony-xl-v5-sdxl',
    'John6666/wai-ani-nsfw-ponyxl-v10-sdxl',
    'John6666/wai-real-mix-v11-sdxl',
    'John6666/wai-shuffle-pdxl-v2-sdxl',
    'John6666/wai-c-v6-sdxl',
    'John6666/iniverse-mix-xl-sfwnsfw-pony-guofeng-v43-sdxl',
    'John6666/sifw-annihilation-xl-v2-sdxl',
    'John6666/photo-realistic-pony-v5-sdxl',
    'John6666/pony-realism-v21main-sdxl',
    'John6666/pony-realism-v22main-sdxl',
    'John6666/cyberrealistic-pony-v63-sdxl',
    'John6666/cyberrealistic-pony-v64-sdxl',
    'John6666/cyberrealistic-pony-v65-sdxl',
    'John6666/cyberrealistic-pony-v7-sdxl',
    'GraydientPlatformAPI/realcartoon-pony-diffusion',
    'John6666/nova-anime-xl-pony-v5-sdxl',
    'John6666/autismmix-sdxl-autismmix-pony-sdxl',
    'John6666/aimz-dream-real-pony-mix-v3-sdxl',
    'John6666/prefectious-xl-nsfw-v10-sdxl',
    'GraydientPlatformAPI/iniverseponyRealGuofeng49',
    'John6666/duchaiten-pony-real-v11fix-sdxl',
    'John6666/duchaiten-pony-real-v20-sdxl',
    'John6666/duchaiten-pony-xl-no-score-v70-sdxl',
    'Spestly/OdysseyXL-3.0',
    'Spestly/OdysseyXL-4.0',
    'KBlueLeaf/Kohaku-XL-Zeta',
    'cagliostrolab/animagine-xl-3.1',
    'yodayo-ai/kivotos-xl-2.0',
    'yodayo-ai/holodayo-xl-2.1',
    'yodayo-ai/clandestine-xl-1.0',
    'digiplay/majicMIX_sombre_v2',
    'digiplay/majicMIX_realistic_v6',
    'digiplay/majicMIX_realistic_v7',
    'digiplay/DreamShaper_8',
    'digiplay/BeautifulArt_v1',
    'digiplay/DarkSushi2.5D_v1',
    'digiplay/darkphoenix3D_v1.1',
    'digiplay/BeenYouLiteL11_diffusers',
    'GraydientPlatformAPI/rev-animated2',
    'myxlmynx/cyberrealistic_classic40',
    'GraydientPlatformAPI/cyberreal6',
    'GraydientPlatformAPI/cyberreal5',
    'youknownothing/deliberate-v6',
    'GraydientPlatformAPI/deliberate-cyber3',
    'GraydientPlatformAPI/picx-real',
    'GraydientPlatformAPI/perfectworld6',
    'emilianJR/epiCRealism',
    'votepurchase/counterfeitV30_v30',
    'votepurchase/ChilloutMix',
    'Meina/MeinaMix_V11',
    'Meina/MeinaUnreal_V5',
    'Meina/MeinaPastel_V7',
    'GraydientPlatformAPI/realcartoon3d-17',
    'GraydientPlatformAPI/realcartoon-pixar11',
    'GraydientPlatformAPI/realcartoon-real17',
    'nitrosocke/Ghibli-Diffusion',
]

DIFFUSERS_FORMAT_LORAS = [
    "nerijs/animation2k-flux",
    "XLabs-AI/flux-RealismLora",
    "Shakker-Labs/FLUX.1-dev-LoRA-Logo-Design",
]

DOWNLOAD_EMBEDS = [
    'https://huggingface.co/datasets/Nerfgun3/bad_prompt/blob/main/bad_prompt_version2.pt',
    # 'https://huggingface.co/embed/negative/resolve/main/EasyNegativeV2.safetensors',
    # 'https://huggingface.co/embed/negative/resolve/main/bad-hands-5.pt',
]

CIVITAI_API_KEY = os.environ.get("CIVITAI_API_KEY")
HF_TOKEN = os.environ.get("HF_READ_TOKEN")

DIRECTORY_MODELS = 'models'
DIRECTORY_LORAS = 'loras'
DIRECTORY_VAES = 'vaes'
DIRECTORY_EMBEDS = 'embedings'
DIRECTORY_UPSCALERS = 'upscalers'

CACHE_HF = "/home/user/.cache/huggingface/hub/"
STORAGE_ROOT = "/home/user/"

TASK_STABLEPY = {
    'txt2img': 'txt2img',
    'img2img': 'img2img',
    'inpaint': 'inpaint',
    # 'canny T2I Adapter': 'sdxl_canny_t2i',  # NO HAVE STEP CALLBACK PARAMETERS SO NOT WORKS WITH DIFFUSERS 0.29.0
    # 'sketch  T2I Adapter': 'sdxl_sketch_t2i',
    # 'lineart  T2I Adapter': 'sdxl_lineart_t2i',
    # 'depth-midas  T2I Adapter': 'sdxl_depth-midas_t2i',
    # 'openpose  T2I Adapter': 'sdxl_openpose_t2i',
    'openpose ControlNet': 'openpose',
    'canny ControlNet': 'canny',
    'mlsd ControlNet': 'mlsd',
    'scribble ControlNet': 'scribble',
    'softedge ControlNet': 'softedge',
    'segmentation ControlNet': 'segmentation',
    'depth ControlNet': 'depth',
    'normalbae ControlNet': 'normalbae',
    'lineart ControlNet': 'lineart',
    'lineart_anime ControlNet': 'lineart_anime',
    'shuffle ControlNet': 'shuffle',
    'ip2p ControlNet': 'ip2p',
    'optical pattern ControlNet': 'pattern',
    'recolor ControlNet': 'recolor',
    'tile ControlNet': 'tile',
    'repaint ControlNet': 'repaint',
}

TASK_MODEL_LIST = list(TASK_STABLEPY.keys())

UPSCALER_DICT_GUI = {
    None: None,
    **{bu: bu for bu in ALL_BUILTIN_UPSCALERS if bu not in ["HAT x4", "DAT x4", "DAT x3", "DAT x2", "SwinIR 4x"]},
    # "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "RealESRNet_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
    # "RealESRGAN_x4plus_anime_6B": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    # "RealESRGAN_x2plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    # "realesr-animevideov3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
    # "realesr-general-x4v3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
    # "realesr-general-wdn-x4v3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
    "4x-UltraSharp": "https://huggingface.co/Shandypur/ESRGAN-4x-UltraSharp/resolve/main/4x-UltraSharp.pth",
    "4x_foolhardy_Remacri": "https://huggingface.co/FacehugmanIII/4x_foolhardy_Remacri/resolve/main/4x_foolhardy_Remacri.pth",
    "Remacri4xExtraSmoother": "https://huggingface.co/hollowstrawberry/upscalers-backup/resolve/main/ESRGAN/Remacri%204x%20ExtraSmoother.pth",
    "AnimeSharp4x": "https://huggingface.co/hollowstrawberry/upscalers-backup/resolve/main/ESRGAN/AnimeSharp%204x.pth",
    "lollypop": "https://huggingface.co/hollowstrawberry/upscalers-backup/resolve/main/ESRGAN/lollypop.pth",
    "RealisticRescaler4x": "https://huggingface.co/hollowstrawberry/upscalers-backup/resolve/main/ESRGAN/RealisticRescaler%204x.pth",
    "NickelbackFS4x": "https://huggingface.co/hollowstrawberry/upscalers-backup/resolve/main/ESRGAN/NickelbackFS%204x.pth"
}

UPSCALER_KEYS = list(UPSCALER_DICT_GUI.keys())

DIFFUSERS_CONTROLNET_MODEL = [
    "Automatic",

    "brad-twinkl/controlnet-union-sdxl-1.0-promax",
    "xinsir/controlnet-union-sdxl-1.0",
    "xinsir/anime-painter",
    "Eugeoter/noob-sdxl-controlnet-canny",
    "Eugeoter/noob-sdxl-controlnet-lineart_anime",
    "Eugeoter/noob-sdxl-controlnet-depth",
    "Eugeoter/noob-sdxl-controlnet-normal",
    "Eugeoter/noob-sdxl-controlnet-softedge_hed",
    "Eugeoter/noob-sdxl-controlnet-scribble_pidinet",
    "Eugeoter/noob-sdxl-controlnet-scribble_hed",
    "Eugeoter/noob-sdxl-controlnet-manga_line",
    "Eugeoter/noob-sdxl-controlnet-lineart_realistic",
    "Eugeoter/noob-sdxl-controlnet-depth_midas-v1-1",
    "dimitribarbot/controlnet-openpose-sdxl-1.0-safetensors",
    "r3gm/controlnet-openpose-sdxl-1.0-fp16",
    "r3gm/controlnet-canny-scribble-integrated-sdxl-v2-fp16",
    "r3gm/controlnet-union-sdxl-1.0-fp16",
    "r3gm/controlnet-lineart-anime-sdxl-fp16",
    "r3gm/control_v1p_sdxl_qrcode_monster_fp16",
    "r3gm/controlnet-tile-sdxl-1.0-fp16",
    "r3gm/controlnet-recolor-sdxl-fp16",
    "r3gm/controlnet-openpose-twins-sdxl-1.0-fp16",
    "r3gm/controlnet-qr-pattern-sdxl-fp16",
    "Yakonrus/SDXL_Controlnet_Tile_Realistic_v2",
    "TheMistoAI/MistoLine",
    "briaai/BRIA-2.3-ControlNet-Recoloring",
    "briaai/BRIA-2.3-ControlNet-Canny",

    "lllyasviel/control_v11p_sd15_openpose",
    "lllyasviel/control_v11p_sd15_canny",
    "lllyasviel/control_v11p_sd15_mlsd",
    "lllyasviel/control_v11p_sd15_scribble",
    "lllyasviel/control_v11p_sd15_softedge",
    "lllyasviel/control_v11p_sd15_seg",
    "lllyasviel/control_v11f1p_sd15_depth",
    "lllyasviel/control_v11p_sd15_normalbae",
    "lllyasviel/control_v11p_sd15_lineart",
    "lllyasviel/control_v11p_sd15s2_lineart_anime",
    "lllyasviel/control_v11e_sd15_shuffle",
    "lllyasviel/control_v11e_sd15_ip2p",
    "lllyasviel/control_v11p_sd15_inpaint",
    "monster-labs/control_v1p_sd15_qrcode_monster",
    "lllyasviel/control_v11f1e_sd15_tile",
    "latentcat/control_v1p_sd15_brightness",
    "yuanqiuye/qrcode_controlnet_v3",

    "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro",
    # "Shakker-Labs/FLUX.1-dev-ControlNet-Pose",
    # "Shakker-Labs/FLUX.1-dev-ControlNet-Depth",
    # "jasperai/Flux.1-dev-Controlnet-Upscaler",
    # "jasperai/Flux.1-dev-Controlnet-Depth",
    # "jasperai/Flux.1-dev-Controlnet-Surface-Normals",
    # "XLabs-AI/flux-controlnet-canny-diffusers",
    # "XLabs-AI/flux-controlnet-hed-diffusers",
    # "XLabs-AI/flux-controlnet-depth-diffusers",
    # "InstantX/FLUX.1-dev-Controlnet-Union",
    # "InstantX/FLUX.1-dev-Controlnet-Canny",
]

PROMPT_W_OPTIONS = [
    ("Compel format: (word)weight", "Compel"),
    ("Classic format: (word:weight)", "Classic"),
    ("Classic-original format: (word:weight)", "Classic-original"),
    ("Classic-no_norm format: (word:weight)", "Classic-no_norm"),
    ("Classic-sd_embed format: (word:weight)", "Classic-sd_embed"),
    ("Classic-ignore", "Classic-ignore"),
    ("None", "None"),
]

WARNING_MSG_VAE = (
    "Use the right VAE for your model to maintain image quality. The wrong"
    " VAE can lead to poor results, like blurriness in the generated images."
)

SDXL_TASK = [k for k, v in TASK_STABLEPY.items() if v in SDXL_TASKS]
SD_TASK = [k for k, v in TASK_STABLEPY.items() if v in SD15_TASKS]
FLUX_TASK = list(TASK_STABLEPY.keys())[:3] + [k for k, v in TASK_STABLEPY.items() if v in FLUX_CN_UNION_MODES.keys()]

MODEL_TYPE_TASK = {
    "SD 1.5": SD_TASK,
    "SDXL": SDXL_TASK,
    "FLUX": FLUX_TASK,
}

MODEL_TYPE_CLASS = {
    "diffusers:StableDiffusionPipeline": "SD 1.5",
    "diffusers:StableDiffusionXLPipeline": "SDXL",
    "diffusers:FluxPipeline": "FLUX",
}

DIFFUSECRAFT_CHECKPOINT_NAME = {
    "sd1.5": "SD 1.5",
    "sdxl": "SDXL",
    "flux-dev": "FLUX",
    "flux-schnell": "FLUX",
}

POST_PROCESSING_SAMPLER = ["Use same sampler"] + [
    name_s for name_s in scheduler_names if "Auto-Loader" not in name_s
]

IP_MODELS = []
ALL_IPA = sorted(set(IP_ADAPTERS_SD + IP_ADAPTERS_SDXL))

for origin_name in ALL_IPA:
    suffixes = []
    if origin_name in IP_ADAPTERS_SD:
        suffixes.append("sd1.5")
    if origin_name in IP_ADAPTERS_SDXL:
        suffixes.append("sdxl")
    ref_name = f"{origin_name} ({'/'.join(suffixes)})"
    IP_MODELS.append((ref_name, origin_name))

MODE_IP_OPTIONS = ["original", "style", "layout", "style+layout"]

SUBTITLE_GUI = (
    "### This demo uses [diffusers](https://github.com/huggingface/diffusers)"
    " to perform different tasks in image generation."
)

HELP_GUI = (
    """### Help:
    - The current space runs on a ZERO GPU which is assigned for approximately 60 seconds; Therefore, if you submit expensive tasks, the operation may be canceled upon reaching the maximum allowed time with 'GPU TASK ABORTED'.
    - Distorted or strange images often result from high prompt weights, so it's best to use low weights and scales, and consider using Classic variants like 'Classic-original'.
    - For better results with Pony Diffusion, try using sampler DPM++ 1s or DPM2 with Compel or Classic prompt weights.
    """
)

EXAMPLES_GUI_HELP = (
    """### The following examples perform specific tasks:
    1. Generation with SDXL and upscale
    2. Generation with FLUX dev
    3. ControlNet Canny SDXL
    4. Optical pattern (Optical illusion) SDXL
    5. Convert an image to a coloring drawing
    6. V prediction model inference
    7. V prediction model sd_embed variant inference
    8. ControlNet OpenPose SD 1.5 and Latent upscale

    - Different tasks can be performed, such as img2img or using the IP adapter, to preserve a person's appearance or a specific style based on an image.
    """
)

EXAMPLES_GUI = [
    [
        "splatter paint theme, 1girl, frame center, pretty face, face with artistic paint artwork, feminism, long hair, upper body view, futuristic expression illustrative painted background, origami, stripes, explosive paint splashes behind her, hand on cheek pose, strobe lighting, masterpiece photography creative artwork, golden morning light, highly detailed, masterpiece, best quality, very aesthetic, absurdres",
        "logo, artist name, (worst quality, normal quality), bad-artist, ((bad anatomy)), ((bad hands)), ((bad proportions)), ((duplicate limbs)), ((fused limbs)), ((interlocking fingers)), ((poorly drawn face)), high contrast., score_6, score_5, score_4, lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]",
        28,
        5.0,
        -1,
        "None",
        0.33,
        "DPM++ 2M SDE",
        1152,
        896,
        "John6666/noobai-xl-nai-xl-epsilonpred10version-sdxl",
        "txt2img",
        "image.webp",  # img conttol
        1024,  # img resolution
        0.35,  # strength
        1.0,  # cn scale
        0.0,  # cn start
        1.0,  # cn end
        "Classic-no_norm",
        "Nearest",
        45,
        False,
    ],
    [
        "a digital illustration of a movie poster titled 'Finding Emo', finding nemo parody poster, featuring a depressed cartoon clownfish with black emo hair, eyeliner, and piercings, bored expression, swimming in a dark underwater scene, in the background, movie title in a dripping, grungy font, moody blue and purple color palette",
        "",
        24,
        3.5,
        -1,
        "None",
        0.33,
        "FlowMatch Euler",
        1152,
        896,
        "black-forest-labs/FLUX.1-dev",
        "txt2img",
        None,  # img conttol
        1024,  # img resolution
        0.35,  # strength
        1.0,  # cn scale
        0.0,  # cn start
        1.0,  # cn end
        "Classic",
        None,
        70,
        True,
    ],
    [
        "((masterpiece)), best quality, blonde disco girl, detailed face, realistic face, realistic hair, dynamic pose, pink pvc, intergalactic disco background, pastel lights, dynamic contrast, airbrush, fine detail, 70s vibe, midriff",
        "(worst quality:1.2), (bad quality:1.2), (poor quality:1.2), (missing fingers:1.2), bad-artist-anime, bad-artist, bad-picture-chill-75v",
        48,
        3.5,
        -1,
        "None",
        0.33,
        "DPM++ 2M SDE Ef",
        1024,
        1024,
        "John6666/epicrealism-xl-v10kiss2-sdxl",
        "canny ControlNet",
        "image.webp",  # img conttol
        1024,  # img resolution
        0.35,  # strength
        1.0,  # cn scale
        0.0,  # cn start
        1.0,  # cn end
        "Classic",
        None,
        44,
        False,
    ],
    [
        "cinematic scenery old city ruins",
        "(worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), (illustration, 3d, 2d, painting, cartoons, sketch, blurry, film grain, noise), (low quality, worst quality:1.2)",
        50,
        4.0,
        -1,
        "None",
        0.33,
        "Euler a",
        1024,
        1024,
        "SG161222/RealVisXL_V5.0",
        "optical pattern ControlNet",
        "spiral_no_transparent.png",  # img conttol
        1024,  # img resolution
        0.35,  # strength
        1.0,  # cn scale
        0.05,  # cn start
        0.8,  # cn end
        "Classic",
        None,
        35,
        False,
    ],
    [
        "black and white, line art, coloring drawing, clean line art, black strokes, no background, white, black, free lines, black scribbles, on paper, A blend of comic book art and lineart full of black and white color, masterpiece, high-resolution, trending on Pixiv fan box, palette knife, brush strokes, two-dimensional, planar vector, T-shirt design, stickers, and T-shirt design, vector art, fantasy art, Adobe Illustrator, hand-painted, digital painting, low polygon, soft lighting, aerial view, isometric style, retro aesthetics, 8K resolution, black sketch lines, monochrome, invert color",
        "color, red, green, yellow, colored, duplicate, blurry, abstract, disfigured, deformed, animated, toy, figure, framed, 3d, bad art, poorly drawn, extra limbs, close up, b&w, weird colors, blurry, watermark, blur haze, 2 heads, long neck, watermark, elongated body, cropped image, out of frame, draft, deformed hands, twisted fingers, double image, malformed hands, multiple heads, extra limb, ugly, poorly drawn hands, missing limb, cut-off, over satured, grain, lowères, bad anatomy, poorly drawn face, mutation, mutated, floating limbs, disconnected limbs, out of focus, long body, disgusting, extra fingers, groos proportions, missing arms, mutated hands, cloned face, missing legs, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draft, deformed, blurry, bad anatomy, disfigured, poorly drawn face, mutation, bluelish, blue",
        20,
        4.0,
        -1,
        "loras/Coloring_book_-_LineArt.safetensors",
        1.0,
        "DPM++ 2M SDE",
        1024,
        1024,
        "eienmojiki/Anything-XL",
        "lineart ControlNet",
        "color_image.png",  # img conttol
        896,  # img resolution
        0.35,  # strength
        1.0,  # cn scale
        0.0,  # cn start
        1.0,  # cn end
        "Compel",
        None,
        35,
        False,
    ],
    [
        "[mochizuki_shiina], [syuri22], newest, reimu, solo, outdoors, water, flower, lantern",
        "worst quality, normal quality, old, sketch,",
        28,
        7.0,
        -1,
        "None",
        0.33,
        "DPM 3M Ef",
        1600,
        1024,
        "Laxhar/noobai-XL-Vpred-1.0",
        "txt2img",
        "color_image.png",  # img conttol
        1024,  # img resolution
        0.35,  # strength
        1.0,  # cn scale
        0.0,  # cn start
        1.0,  # cn end
        "Classic",
        None,
        30,
        False,
    ],
    [
        "[mochizuki_shiina], [syuri22], newest, multiple girls, 2girls, earrings, jewelry, gloves, purple eyes, black hair, looking at viewer, nail polish, hat, smile, open mouth, fingerless gloves, sleeveless, :d, upper body, blue eyes, closed mouth, black gloves, hands up, long hair, shirt, bare shoulders, white headwear, blush, black headwear, blue nails, upper teeth only, short hair, white gloves, white shirt, teeth, rabbit hat, star earrings, purple nails, pink hair, detached sleeves, fingernails, fake animal ears, animal hat, sleeves past wrists, black shirt, medium hair, fur trim, sleeveless shirt, turtleneck, long sleeves, rabbit ears, star \\(symbol\\)",
        "worst quality, normal quality, old, sketch,",
        28,
        7.0,
        -1,
        "None",
        0.33,
        "DPM 3M Ef",
        1600,
        1024,
        "Laxhar/noobai-XL-Vpred-1.0",
        "txt2img",
        "color_image.png",  # img conttol
        1024,  # img resolution
        0.35,  # strength
        1.0,  # cn scale
        0.0,  # cn start
        1.0,  # cn end
        "Classic-sd_embed",
        None,
        30,
        False,
    ],
    [
        "1girl,face,curly hair,red hair,white background,",
        "(worst quality:2),(low quality:2),(normal quality:2),lowres,watermark,",
        38,
        5.0,
        -1,
        "None",
        0.33,
        "DPM++ 2M SDE",
        512,
        512,
        "digiplay/majicMIX_realistic_v7",
        "openpose ControlNet",
        "image.webp",  # img conttol
        1024,  # img resolution
        0.35,  # strength
        1.0,  # cn scale
        0.0,  # cn start
        0.9,  # cn end
        "Classic-original",
        "Latent (antialiased)",
        46,
        False,
    ],
]

RESOURCES = (
    """### Resources
    - John6666's space has some great features you might find helpful [link](https://huggingface.co/spaces/John6666/DiffuseCraftMod).
    - You can also try the image generator in Colab’s free tier, which provides free GPU [link](https://github.com/R3gm/SD_diffusers_interactive).
    """
)