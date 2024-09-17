import gradio as gr
from diffusers import DiffusionPipeline
import os
import torch
import shutil
import spaces


def find_cuda():
    # Check if CUDA_HOME or CUDA_PATH environment variables are set
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')

    if cuda_home and os.path.exists(cuda_home):
        return cuda_home

    # Search for the nvcc executable in the system's PATH
    nvcc_path = shutil.which('nvcc')

    if nvcc_path:
        # Remove the 'bin/nvcc' part to get the CUDA installation path
        cuda_path = os.path.dirname(os.path.dirname(nvcc_path))
        return cuda_path

    return None


cuda_path = find_cuda()

if cuda_path:
    print(f"CUDA installation found at: {cuda_path}")
else:
    print("CUDA installation not found")

# check if cuda is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# load the pipeline/model
pipeline = DiffusionPipeline.from_pretrained("jadechoghari/mar", trust_remote_code=True,
                                             custom_pipeline="jadechoghari/mar")


# function that generates images
@spaces.GPU
def generate_image(seed, num_ar_steps, class_labels, cfg_scale, cfg_schedule):
    generated_image = pipeline(
        model_type="mar_huge",  # using mar_huge
        seed=seed,
        num_ar_steps=num_ar_steps,
        class_labels=[int(label.strip()) for label in class_labels.split(',')],
        cfg_scale=cfg_scale,
        cfg_schedule=cfg_schedule,
        output_dir="./images"
    )
    return generated_image


with gr.Blocks() as demo:
    gr.Markdown("""
    # MAR Image Generation Demo 🚀

    Welcome to the demo for **MAR** (Masked Autoregressive Model), a novel approach to image generation that eliminates the need for vector quantization. MAR uses a diffusion process to generate images in a continuous-valued space, resulting in faster, more efficient, and higher-quality outputs.

    Simply adjust the parameters below to create your custom images in real-time.

    Make sure to provide valid **ImageNet class labels** to see the translation of text to image. For a complete list of ImageNet classes, check out [this reference](https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/). 

    For more details, visit the [GitHub repository](https://github.com/LTH14/mar).
    """)

    seed = gr.Number(value=0, label="Seed")
    num_ar_steps = gr.Slider(minimum=1, maximum=256, value=64, label="Number of AR Steps")
    class_labels = gr.Textbox(value="207, 360, 388, 113, 355, 980, 323, 979",
                              label="Class Labels (comma-separated ImageNet labels)")
    cfg_scale = gr.Slider(minimum=1, maximum=10, value=4, label="CFG Scale")
    cfg_schedule = gr.Dropdown(choices=["constant", "linear"], label="CFG Schedule", value="constant")

    image_output = gr.Image(label="Generated Image")

    generate_button = gr.Button("Generate Image")

    # we link the button to the function and display the output
    generate_button.click(generate_image, inputs=[seed, num_ar_steps, class_labels, cfg_scale, cfg_schedule],
                          outputs=image_output)

    gr.Interface(
        generate_image,
        inputs=[seed, num_ar_steps, class_labels, cfg_scale, cfg_schedule],
        outputs=image_output,
    )

demo.launch()