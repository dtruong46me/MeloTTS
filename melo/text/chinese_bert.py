"""Module for extracting BERT features from Chinese text for TTS."""

import sys
from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM


# model_id = 'hfl/chinese-roberta-wwm-ext-large'
local_path = "./bert/chinese-roberta-wwm-ext-large"


tokenizers = {}
models = {}

def get_bert_feature(
    text: str,
    word2ph: List[int],
    device: Optional[str] = None,
    model_id: str = "hfl/chinese-roberta-wwm-ext-large",
) -> torch.Tensor:
    """Get BERT features for the given text and align them with phonemes.

    Args:
        text (str): The input text.
        word2ph (List[int]): A list mapping words to their corresponding number of phonemes.
        device (Optional[str], optional): The device to use for inference. Defaults to None.
        model_id (str, optional): The ID of the BERT model to use. Defaults to 'hfl/chinese-roberta-wwm-ext-large'.

    Returns:
        torch.Tensor: A tensor containing the phone-level BERT features.
    """
    if model_id not in models:
        models[model_id] = AutoModelForMaskedLM.from_pretrained(
            model_id
        ).to(device)
        tokenizers[model_id] = AutoTokenizer.from_pretrained(model_id)
    model = models[model_id]
    tokenizer = tokenizers[model_id]

    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    if not device:
        device = "cuda"

    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
    # import pdb; pdb.set_trace()
    # assert len(word2ph) == len(text) + 2
    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T


if __name__ == "__main__":
    import torch

    word_level_feature = torch.rand(38, 1024)  # 12个词,每个词1024维特征
    word2phone = [
        1,
        2,
        1,
        2,
        2,
        1,
        2,
        2,
        1,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        2,
        1,
        1,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        1,
    ]

    # 计算总帧数
    total_frames = sum(word2phone)
    print(word_level_feature.shape)
    print(word2phone)
    phone_level_feature = []
    for i in range(len(word2phone)):
        print(word_level_feature[i].shape)

        # 对每个词重复word2phone[i]次
        repeat_feature = word_level_feature[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    print(phone_level_feature.shape)  # torch.Size([36, 1024])
