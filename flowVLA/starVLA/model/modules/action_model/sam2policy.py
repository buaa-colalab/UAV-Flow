import torch
from transformers import Sam2Processor, Sam2Model
import torch
from typing import Optional
import math
from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformers.utils.generic import OutputRecorder
from transformers.models.sam2.modeling_sam2 import Sam2ImageSegmentationOutput

# TODO only video model will have meroy attention
# from transformers import Sam2VideoModel, Sam2VideoProcessor
# import torch
# Modified point_prompt embeds in SAM2

class SAM2Action:
    def __init__(self, model_name="playground/Pretrained_models/sam2-hiera-large", device="cuda"):

        self.device = device if torch.cuda.is_available() else "cpu"
        self.sam2model = Sam2Model.from_pretrained(model_name).to(self.device)
        self.processor = Sam2Processor.from_pretrained(model_name)

    def forward(self, raw_image, input_points=None, input_labels=None):
        
        if input_points is not None and input_labels is not None:
            inputs = self.processor(images=raw_image, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(images=raw_image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.sam2forward(**inputs)
        return outputs

    def sam2forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_points: Optional[torch.FloatTensor] = None,
        input_labels: Optional[torch.LongTensor] = None,
        input_boxes: Optional[torch.FloatTensor] = None,
        input_masks: Optional[torch.LongTensor] = None,
        image_embeddings: Optional[torch.FloatTensor] = None,
        multimask_output: bool = True,
        attention_similarity: Optional[torch.FloatTensor] = None,
        target_embedding: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        r"""
        input_points (`torch.FloatTensor` of shape `(batch_size, num_points, 2)`):
            Input 2D spatial points, this is used by the prompt encoder to encode the prompt. Generally yields to much
            better results. The points can be obtained by passing a list of list of list to the processor that will
            create corresponding `torch` tensors of dimension 4. The first dimension is the image batch size, the
            second dimension is the point batch size (i.e. how many segmentation masks do we want the model to predict
            per input point), the third dimension is the number of points per segmentation mask (it is possible to pass
            multiple points for a single mask), and the last dimension is the x (vertical) and y (horizontal)
            coordinates of the point. If a different number of points is passed either for each image, or for each
            mask, the processor will create "PAD" points that will correspond to the (0, 0) coordinate, and the
            computation of the embedding will be skipped for these points using the labels.
        input_labels (`torch.LongTensor` of shape `(batch_size, point_batch_size, num_points)`):
            Input labels for the points, this is used by the prompt encoder to encode the prompt. According to the
            official implementation, there are 3 types of labels

            - `1`: the point is a point that contains the object of interest
            - `0`: the point is a point that does not contain the object of interest
            - `-1`: the point corresponds to the background

            We added the label:

            - `-10`: the point is a padding point, thus should be ignored by the prompt encoder

            The padding labels should be automatically done by the processor.
        input_boxes (`torch.FloatTensor` of shape `(batch_size, num_boxes, 4)`):
            Input boxes for the points, this is used by the prompt encoder to encode the prompt. Generally yields to
            much better generated masks. The boxes can be obtained by passing a list of list of list to the processor,
            that will generate a `torch` tensor, with each dimension corresponding respectively to the image batch
            size, the number of boxes per image and the coordinates of the top left and bottom right point of the box.
            In the order (`x1`, `y1`, `x2`, `y2`):

            - `x1`: the x coordinate of the top left point of the input box
            - `y1`: the y coordinate of the top left point of the input box
            - `x2`: the x coordinate of the bottom right point of the input box
            - `y2`: the y coordinate of the bottom right point of the input box
        input_masks (`torch.FloatTensor` of shape `(batch_size, image_size, image_size)`):
            SAM model also accepts segmentation masks as input. The mask will be embedded by the prompt encoder to
            generate a corresponding embedding, that will be fed later on to the mask decoder. These masks needs to be
            manually fed by the user, and they need to be of shape (`batch_size`, `image_size`, `image_size`).
        image_embeddings (`torch.FloatTensor` of shape `(batch_size, output_channels, window_size, window_size)`):
            Image embeddings, this is used by the mask decoder to generate masks and iou scores. For more memory
            efficient computation, users can first retrieve the image embeddings using the `get_image_embeddings`
            method, and then feed them to the `forward` method instead of feeding the `pixel_values`.
        multimask_output (`bool`, *optional*):
            In the original implementation and paper, the model always outputs 3 masks per image (or per point / per
            bounding box if relevant). However, it is possible to just output a single mask, that corresponds to the
            "best" mask, by specifying `multimask_output=False`.
        attention_similarity (`torch.FloatTensor`, *optional*):
            Attention similarity tensor, to be provided to the mask decoder for target-guided attention in case the
            model is used for personalization as introduced in [PerSAM](https://huggingface.co/papers/2305.03048).
        target_embedding (`torch.FloatTensor`, *optional*):
            Embedding of the target concept, to be provided to the mask decoder for target-semantic prompting in case
            the model is used for personalization as introduced in [PerSAM](https://huggingface.co/papers/2305.03048).

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoModel, AutoProcessor

        >>> model = AutoModel.from_pretrained("danelcsb/sam2.1_hiera_tiny")
        >>> processor = AutoProcessor.from_pretrained("danelcsb/sam2.1_hiera_tiny")

        >>> img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/sam-car.png"
        >>> raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
        >>> input_points = [[[400, 650]]]  # 2D location of a window on the car
        >>> inputs = processor(images=raw_image, input_points=input_points, return_tensors="pt")

        >>> # Get segmentation mask
        >>> outputs = model(**inputs)

        >>> # Postprocess masks
        >>> masks = processor.post_process_masks(
        ...     outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
        ... )
        ```
        """
        if not ((pixel_values is None) ^ (image_embeddings is None)):
            raise ValueError("Exactly one of pixel_values or image_embeddings must be provided.")
        if input_points is not None and input_boxes is not None:
            if input_points.shape[1] != input_boxes.shape[1]:
                raise ValueError(
                    f"You should provide as many bounding boxes as input points per box. Got {input_points.shape[1]} and {input_boxes.shape[1]}."
                )

        image_positional_embeddings = self.sam2model.get_image_wide_positional_embeddings()
        # repeat with batch size
        batch_size = pixel_values.shape[0] if pixel_values is not None else image_embeddings[-1].shape[0]
        image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)

        vision_attentions = None
        vision_hidden_states = None

        if pixel_values is not None:
            feature_maps, _, vision_hidden_states, vision_attentions = self.get_image_features(
                pixel_values,
                **kwargs,
            )

            # add no memory embedding to the last feature map
            feature_maps[-1] = feature_maps[-1] + self.sam2model.no_memory_embedding

            # reshape feature maps to the same shape as the backbone feature sizes
            image_embeddings = [
                feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
                for feat, feat_size in zip(feature_maps, self.sam2model.backbone_feature_sizes) # multi-scale ViT
            ]

        if input_points is not None and input_labels is None:
            input_labels = torch.ones_like(input_points[:, :, :, 0], dtype=torch.int, device=input_points.device)

        if input_points is None and input_boxes is None:
            # If no points are provide, pad with an empty point (with label -1)
            input_points = torch.zeros(
                batch_size, 1, 1, 2, dtype=image_embeddings[-1].dtype, device=image_embeddings[-1].device
            )
            input_labels = -torch.ones(batch_size, 1, 1, dtype=torch.int32, device=image_embeddings[-1].device)

        if input_masks is not None:
            # If mask_inputs is provided, downsize it into low-res mask input if needed
            # and feed it as a dense mask prompt into the SAM mask encoder
            if input_masks.shape[-2:] != self.sam2model.prompt_encoder.mask_input_size:
                input_masks = F.interpolate(
                    input_masks.float(),
                    size=self.sam2model.prompt_encoder.mask_input_size,
                    align_corners=False,
                    mode="bilinear",
                    antialias=True,  # use antialias for downsampling
                ).to(input_masks.dtype)
        #[B, 1, 2, 256], [B, 256, H, W]
        sparse_embeddings, dense_embeddings = self.sam2model.sam2model.prompt_encoder(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            input_masks=input_masks,
        )
        low_res_multimasks, iou_scores, _, object_score_logits = self.sam2model.sam2model.mask_decoder(
            image_embeddings=image_embeddings[-1], # last_level feature map
            image_positional_embeddings=image_positional_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            high_resolution_features=image_embeddings[:-1],
            attention_similarity=attention_similarity,
            target_embedding=target_embedding,
            **kwargs,
        )

        return Sam2ImageSegmentationOutput(
            iou_scores=iou_scores,
            pred_masks=low_res_multimasks,
            object_score_logits=object_score_logits,
            image_embeddings=image_embeddings,
            vision_hidden_states=vision_hidden_states,
            vision_attentions=vision_attentions,
        )
    

    def segment(self, raw_image, input_points=None, input_labels=None):

        outputs = self.forward(raw_image, input_points, input_labels)

        masks = self.processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]

        return masks



def show_results(masks, raw_image, output_path="assets/bar/masks_overlay.png"):

    import matplotlib.pyplot as plt
    import numpy as np

    # 画图保存查看 masks
    # masks.shape torch.Size([1, 3, 1200, 1800])
    # raw_image <PIL.Image.Image image mode=RGB size=1800x1200 at 0x7FA1D9FB0DF0>

    # 将 masks 转换为 numpy 数组
    masks_np = masks.squeeze(0).numpy()  # shape: (3, 1200, 1800)

    # 创建一个叠加的掩码图像
    overlay = np.zeros((masks_np.shape[1], masks_np.shape[2], 3), dtype=np.uint8)

    # 为每个掩码分配不同的颜色
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # 红、绿、蓝
    for i, color in enumerate(colors):
        overlay[masks_np[i] > 0.5] = color  # 阈值为 0.5

    # 将原始图像和掩码叠加
    raw_image_np = np.array(raw_image)
    combined = (0.7 * raw_image_np + 0.3 * overlay).astype(np.uint8)

    # 保存叠加后的图像
    output_path = "masks_overlay.png"
    Image.fromarray(combined).save(output_path)
    print(f"掩码叠加图像已保存到 {output_path}")

    # 可视化结果
    plt.figure(figsize=(10, 10))
    plt.imshow(combined)
    plt.axis("off")
    plt.title("Masks Overlay")
    # plt.show()
    # save
    plt.imsave(output_path, combined)

if __name__ =="__main__":

    # Basic Image Segmentation

    from omegaconf import OmegaConf
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./starVLA/config/training/internvla_cotrain_custom.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    from transformers import Sam2Processor, Sam2Model
    import torch
    from PIL import Image
    import requests

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam2policy = SAM2Action(device=device)

    # Load multiple images
    image_urls = [
        "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/truck.jpg",
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/dog-sam.png"
    ]
    raw_images = [Image.open(requests.get(url, stream=True).raw).convert("RGB") for url in image_urls]

    # Single point per image
    input_points = [[[[500, 375]]], [[[770, 200]]]]  # One point for each image
    input_labels = [[[1]], [[1]]]  # Positive clicks for both images


    with torch.no_grad():
        outputs = sam2policy(raw_images, input_points, input_labels)



    processor = sam2policy.processor
    inputs = processor(images=raw_images, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(device)

    # Post-process masks for each image
    all_masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])
    print(f"Processed {len(all_masks)} images, each with {all_masks[0].shape[0]} objects")

    # 画图保存查看 masks
    # masks.shape torch.Size([1, 3, 1200, 1800])
    # raw_image <PIL.Image.Image image mode=RGB size=1800x1200 at 0x7FA1D9FB0DF0>

    show_results(all_masks[0], raw_images[0])
