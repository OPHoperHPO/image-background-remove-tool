from pathlib import Path

import click
import tqdm

from carvekit.utils.image_utils import ALLOWED_SUFFIXES
from carvekit.utils.pool_utils import batch_generator, thread_pool_processing
from carvekit.web.schemas.config import MLConfig
from carvekit.web.utils.init_utils import init_interface
from carvekit.utils.fs_utils import save_file


@click.command(
    "removebg",
    help="Performs background removal on specified photos using console interface.",
)
@click.option("-i", required=True, type=str, help="Path to input file or dir")
@click.option("-o", default="none", type=str, help="Path to output file or dir")
@click.option("--pre", default="autoscene", type=str, help="Preprocessing method")
@click.option("--post", default="cascade_fba", type=str, help="Postprocessing method.")
@click.option("--net", default="tracer_b7", type=str, help="Segmentation Network")
@click.option(
    "--recursive",
    default=False,
    type=bool,
    help="Enables recursive search for images in a folder",
)
@click.option(
    "--batch_size",
    default=10,
    type=int,
    help="Batch Size for list of images to be loaded to RAM",
)
@click.option(
    "--batch_size_pre",
    default=5,
    type=int,
    help="Batch size for list of images to be processed by preprocessing method network",
)
@click.option(
    "--batch_size_seg",
    default=5,
    type=int,
    help="Batch size for list of images to be processed by segmentation " "network",
)
@click.option(
    "--batch_size_mat",
    default=1,
    type=int,
    help="Batch size for list of images to be processed by matting " "network",
)
@click.option(
    "--batch_size_refine",
    default=1,
    type=int,
    help="Batch size for list of images to be processed by refining network",
)
@click.option(
    "--seg_mask_size",
    default=640,
    type=int,
    help="The size of the input image for the segmentation neural network.",
)
@click.option(
    "--matting_mask_size",
    default=2048,
    type=int,
    help="The size of the input image for the matting neural network.",
)
@click.option(
    "--refine_mask_size",
    default=900,
    type=int,
    help="The size of the input image for the refining neural network.",
)
@click.option(
    "--trimap_dilation",
    default=30,
    type=int,
    help="The size of the offset radius from the object mask in "
    "pixels when forming an unknown area",
)
@click.option(
    "--trimap_erosion",
    default=5,
    type=int,
    help="The number of iterations of erosion that the object's "
    "mask will be subjected to before forming an unknown area",
)
@click.option(
    "--trimap_prob_threshold",
    default=231,
    type=int,
    help="Probability threshold at which the prob_filter "
    "and prob_as_unknown_area operations will be "
    "applied",
)
@click.option("--device", default="cpu", type=str, help="Processing Device.")
@click.option(
    "--fp16", default=False, type=bool, help="Enables mixed precision processing."
)
def removebg(
    i: str,
    o: str,
    pre: str,
    post: str,
    net: str,
    recursive: bool,
    batch_size: int,
    batch_size_pre: int,
    batch_size_seg: int,
    batch_size_mat: int,
    batch_size_refine: int,
    seg_mask_size: int,
    matting_mask_size: int,
    refine_mask_size: int,
    device: str,
    fp16: bool,
    trimap_dilation: int,
    trimap_erosion: int,
    trimap_prob_threshold: int,
):
    out_path = Path(o)
    input_path = Path(i)
    if input_path.is_dir():
        if recursive:
            all_images = input_path.rglob("*.*")
        else:
            all_images = input_path.glob("*.*")
        all_images = [
            i
            for i in all_images
            if i.suffix.lower() in ALLOWED_SUFFIXES and "_bg_removed" not in i.name
        ]
    else:
        all_images = [input_path]

    interface_config = MLConfig(
        segmentation_network=net,
        preprocessing_method=pre,
        postprocessing_method=post,
        device=device,
        batch_size_seg=batch_size_seg,
        batch_size_matting=batch_size_mat,
        batch_size_refine=batch_size_refine,
        seg_mask_size=seg_mask_size,
        matting_mask_size=matting_mask_size,
        refine_mask_size=refine_mask_size,
        fp16=fp16,
        trimap_dilation=trimap_dilation,
        trimap_erosion=trimap_erosion,
        trimap_prob_threshold=trimap_prob_threshold,
        batch_size_pre=batch_size_pre,
    )

    interface = init_interface(interface_config)

    for image_batch in tqdm.tqdm(
        batch_generator(all_images, n=batch_size),
        total=int(len(all_images) / batch_size),
        desc="Removing background",
        unit=" image batch",
        colour="blue",
    ):
        images_without_background = interface(image_batch)  # Remove background
        thread_pool_processing(
            lambda x: save_file(out_path, image_batch[x], images_without_background[x]),
            range((len(image_batch))),
        )  # Drop images to fs


if __name__ == "__main__":
    removebg()
