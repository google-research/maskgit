import tempfile
import numpy as np
import jax
import jax.numpy as jnp
from timeit import default_timer as timer
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from cog import BasePredictor, Path, Input, BaseModel, File

import maskgit
from maskgit.utils import restore_from_path, draw_image_with_bbox, Bbox
from maskgit.inference import ImageNet_class_conditional_generator


with open("category.txt") as infile:
    CATEGORY = [line.rstrip() for line in infile]


class Predictor(BasePredictor):
    def setup(self):
        self.generator_256 = ImageNet_class_conditional_generator(image_size=256)
        self.generator_512 = ImageNet_class_conditional_generator(image_size=512)

    def predict(
        self,
        task_type: str = Input(
            choices=[
                "Class-conditional Image Synthesis",
                "Class-conditional Image Editing",
            ],
            default="Class-conditional Image Synthesis",
            description="Choose task type.",
        ),
        category: str = Input(
            choices=CATEGORY,
            description="Choose the ImageNet label, which determines what type of object to synthesize or edit to.",
        ),
        image_size: int = Input(
            choices=[256, 512],
            default=256,
            description="Choose the size of the generated image. Output will generate 8 images.",
        ),
        image: Path = Input(
            default=None,
            description="Provide input image for Class-conditional Image Editing. "
            "The image will be resized to image_size.",
        ),
        bbox_top_left_height_width: str = Input(
            default=None,
            description="For Class-conditional Image Editing, provide the area for image editing in the format of "
            "top_left_height_width, e.g. input image 128_64_256_288. Output will show the resized image "
            "with the box highlighting the edited area and 8 edited images.",
        ),
    ) -> Path:

        rng = jax.random.PRNGKey(42)
        rng, sample_rng = jax.random.split(rng)
        label = int(category.split(")")[0])
        generator = self.generator_512 if image_size == 512 else self.generator_256

        out_path = Path(tempfile.mkdtemp()) / "output.png"

        if task_type == "Class-conditional Image Synthesis":
            # prep the input tokens based on the chosen label
            input_tokens = generator.create_input_tokens_normal(label)

            start_timer = timer()

            # use normal mode for replicate demo
            results = generator.generate_samples(input_tokens, sample_rng)
            end_timer = timer()
            print(
                f"generated {generator.eval_batch_size()} images in {end_timer - start_timer} seconds"
            )

            # Visualize
            result_img = get_res_images(results)
            plt.imshow(result_img)
            plt.axis("off")
            plt.savefig(str(out_path), bbox_inches="tight", dpi=600)

        else:
            assert (
                image is not None and bbox_top_left_height_width is not None
            ), "For Class-conditional Image Editing, please provide both input image and the area for editing."

            pil_image = Image.open(str(image)).convert("RGB")
            img_width, img_height = pil_image.size
            pil_image = pil_image.resize((image_size, image_size), Image.BICUBIC)
            img = np.float32(pil_image) / 255.0

            bbox = Bbox(bbox_top_left_height_width)
            assert (
                bbox.top >= 0
                and bbox.left >= 0
                and bbox.top + bbox.height <= img_height
                and bbox.left + bbox.width <= img_width
            ), "The box provided is not the range of the input image."

            # visualize it with our bounding box
            fig, ax = plt.subplots(2, 1, gridspec_kw={"height_ratios": [1, 8]})
            plt.subplots_adjust(hspace=0.05)
            ax[0].imshow(img)
            rect = patches.Rectangle(
                (bbox.left, bbox.top),
                bbox.width,
                bbox.height,
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            ax[0].add_patch(rect)
            ax[0].axis("off")

            (
                latent_mask,
                input_tokens,
            ) = generator.create_latent_mask_and_input_tokens_for_image_editing(
                img, bbox, label
            )

            rng, sample_rng = jax.random.split(rng)

            results = generator.generate_samples(
                input_tokens, sample_rng, start_iter=2, num_iterations=12
            )

            # -----------------------
            # Post-process by applying a gaussian blur using the input
            # and output images.
            composite_images = generator.composite_outputs(img, latent_mask, results)
            # -----------------------
            result_img = get_res_images(composite_images)
            ax[1].imshow(result_img)
            ax[1].axis("off")
            plt.savefig(str(out_path), bbox_inches="tight", dpi=600)
        return out_path


def get_res_images(images):
    batch_size, height, width, c = images.shape
    # images = images.swapaxes(0, 1)
    image_grid = images.reshape(batch_size * height, width, c)
    image_grid = np.clip(image_grid, 0, 1)
    return image_grid
