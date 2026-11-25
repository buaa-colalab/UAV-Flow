import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor
from typing import List, Dict, Any, Optional, Sequence
from PIL import Image
import os
import json

DEFAULT_IMAGE_TOKEN = "<image>"
IGNORE_INDEX = -100


class Qwen3VLDataset(Dataset):
    """
    Dataset that produces samples for Qwen3-VL multimodal training or inference.

    Each sample is a dict like:
    {
        "images": [path1, path2, ...],
        "instruction": "Describe what is happening in the image.",
        "solution": "A man is playing basketball."
    }
    """

    def __init__(self, data: List[Dict[str, Any]], processor: AutoProcessor, with_solution: bool = True, is_debug=False):
        self.data = data

        self.is_debug = is_debug

        self.processor = processor
        self.with_solution = with_solution

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # import ipdb;ipdb.set_trace()

        images = []
        if isinstance(item["image_id"], list):
            for img_path in item["image_id"]:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((256, 256))
                images.append(img)
        else:
            img = Image.open(item["image_id"]).convert("RGB")
            img = img.resize((256, 256))
            images.append(img)

        instruction = item["question"]
        solution = item.get("answer") if self.with_solution else None

        # === Build Qwen3-VL message ===
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": instruction})

        messages = [{"role": "user", "content": content}]

        if solution is not None and not self.is_debug:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})

        # === Convert to model inputs ===
        return {"message": messages, "answer": solution, "image_id": item["image_id"]}


class Qwen3VLDataCollator(object):
    """
    Collator for Qwen3-VL multimodal training.
    """

    def __init__(self, processor: AutoProcessor, with_solution: bool = True, is_debug: bool = False):
        self.processor = processor
        self.with_solution = with_solution
        self.is_debug = is_debug

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        messages = [inst["message"] for inst in instances]

        if self.is_debug:
            print("Image: ", instances[0]["image_id"], "\nAnswer: ", instances[0]["answer"])

        model_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=not self.with_solution or self.is_debug,
            return_tensors="pt",
            return_dict=True,
        )
        # If supervised (has solution), we create labels
        if self.with_solution:
            labels = model_inputs["input_ids"].clone()
            labels[labels == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX

            assistant_tokens = [151644, 77091, 198]

            for i in range(labels.shape[0]):
                seq = labels[i]
                idx = -1
                for j in range(len(seq) - 1):
                    if (
                        seq[j] == assistant_tokens[0]
                        and seq[j + 1] == assistant_tokens[1]
                        and seq[j + 2] == assistant_tokens[2]
                    ):
                        idx = j
                        break
                if idx >= 0:
                    seq[: idx + 3] = IGNORE_INDEX
                else:
                    seq[:] = IGNORE_INDEX

            model_inputs["labels"] = labels

        return model_inputs


def make_qwen3_vlm_dataloader(cfg, is_debug=False):
    """
    Build dataloader that yields Qwen3-VL-ready batches.
    """
    dataset_path = cfg.datasets.vlm_data.dataset_path
    data_list_all = []
    for dataset in dataset_path:
        with open(dataset, "r") as f:
            data_list = json.load(f)
        data_list_all += data_list
    # import ipdb;ipdb.set_trace()
    processor_path = cfg.framework.qwenvl.base_vlm
    processor = AutoProcessor.from_pretrained(processor_path)
    processor.tokenizer.padding_side = "left"
    dataset = Qwen3VLDataset(data_list_all, processor, with_solution=True, is_debug=is_debug)

    data_collator = Qwen3VLDataCollator(processor=processor, is_debug=is_debug)

    batch_size = cfg.datasets.vlm_data.per_device_batch_size
    # num_workers=4
    if is_debug:
        num_workers = 0
    else:
        num_workers = 4

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=not is_debug,
        collate_fn=data_collator,
    )
    # dataloader.batch_size = batch_size
    return dataloader
