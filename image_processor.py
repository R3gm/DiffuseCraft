import spaces
import gradio as gr
from stablepy import Preprocessor

PREPROCESSOR_TASKS_LIST = [
    "Canny",
    "Openpose",
    "DPT",
    "Midas",
    "ZoeDepth",
    "DepthAnything",
    "HED",
    "PidiNet",
    "TEED",
    "Lineart",
    "LineartAnime",
    "Anyline",
    "Lineart standard",
    "SegFormer",
    "UPerNet",
    "ContentShuffle",
    "Recolor",
    "Blur",
    "MLSD",
    "NormalBae",
]

preprocessor = Preprocessor()


def process_inputs(
    image,
    name,
    resolution,
    precessor_resolution,
    low_threshold,
    high_threshold,
    value_threshod,
    distance_threshold,
    recolor_mode,
    recolor_gamma_correction,
    blur_k_size,
    pre_openpose_extra,
    hed_scribble,
    pre_pidinet_safe,
    pre_lineart_coarse,
    use_cuda,
):
    if not image:
        raise ValueError("To use this, simply upload an image.")

    preprocessor.load(name, False)

    params = dict(
        image_resolution=resolution,
        detect_resolution=precessor_resolution,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        thr_v=value_threshod,
        thr_d=distance_threshold,
        mode=recolor_mode,
        gamma_correction=recolor_gamma_correction,
        blur_sigma=blur_k_size,
        hand_and_face=pre_openpose_extra,
        scribble=hed_scribble,
        safe=pre_pidinet_safe,
        coarse=pre_lineart_coarse,
    )

    if use_cuda:
        @spaces.GPU(duration=15)
        def wrapped_func():
            preprocessor.to("cuda")
            return preprocessor(image, **params)
        return wrapped_func()

    return preprocessor(image, **params)


def preprocessor_tab():
    with gr.Row():
        with gr.Column():
            pre_image = gr.Image(label="Image", type="pil", sources=["upload"])
            pre_options = gr.Dropdown(label="Preprocessor", choices=PREPROCESSOR_TASKS_LIST, value=PREPROCESSOR_TASKS_LIST[0])
            pre_img_resolution = gr.Slider(
                minimum=64, maximum=4096, step=64, value=1024, label="Image Resolution",
                info="The maximum proportional size of the generated image based on the uploaded image."
            )
            pre_start = gr.Button(value="PROCESS IMAGE", variant="primary")
            with gr.Accordion("Advanced Settings", open=False):
                with gr.Column():
                    pre_processor_resolution = gr.Slider(minimum=64, maximum=2048, step=64, value=512, label="Preprocessor Resolution")
                    pre_low_threshold = gr.Slider(minimum=1, maximum=255, step=1, value=100, label="'CANNY' low threshold")
                    pre_high_threshold = gr.Slider(minimum=1, maximum=255, step=1, value=200, label="'CANNY' high threshold")
                    pre_value_threshold = gr.Slider(minimum=1, maximum=2.0, step=0.01, value=0.1, label="'MLSD' Hough value threshold")
                    pre_distance_threshold = gr.Slider(minimum=1, maximum=20.0, step=0.01, value=0.1, label="'MLSD' Hough distance threshold")
                    pre_recolor_mode = gr.Dropdown(label="'RECOLOR' mode", choices=["luminance", "intensity"], value="luminance")
                    pre_recolor_gamma_correction = gr.Number(minimum=0., maximum=25., value=1., step=0.001, label="'RECOLOR' gamma correction")
                    pre_blur_k_size = gr.Number(minimum=0, maximum=100, value=9, step=1, label="'BLUR' sigma")
                    pre_openpose_extra = gr.Checkbox(value=True, label="'OPENPOSE' face and hand")
                    pre_hed_scribble = gr.Checkbox(value=False, label="'HED' scribble")
                    pre_pidinet_safe = gr.Checkbox(value=False, label="'PIDINET' safe")
                    pre_lineart_coarse = gr.Checkbox(value=False, label="'LINEART' coarse")
                    pre_use_cuda = gr.Checkbox(value=False, label="Use CUDA")

        with gr.Column():
            pre_result = gr.Image(label="Result", type="pil", interactive=False, format="png")

            pre_start.click(
                fn=process_inputs,
                inputs=[
                    pre_image,
                    pre_options,
                    pre_img_resolution,
                    pre_processor_resolution,
                    pre_low_threshold,
                    pre_high_threshold,
                    pre_value_threshold,
                    pre_distance_threshold,
                    pre_recolor_mode,
                    pre_recolor_gamma_correction,
                    pre_blur_k_size,
                    pre_openpose_extra,
                    pre_hed_scribble,
                    pre_pidinet_safe,
                    pre_lineart_coarse,
                    pre_use_cuda,
                ],
                outputs=[pre_result],
            )
