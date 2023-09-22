# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional

import fire
import numpy as np
from llama import Llama
import json
from tqdm import tqdm
import pandas as pd
import os
from utils.load_arxiv_2023 import get_raw_text_arxiv_2023 as get_raw_text

# CoT
# PROMPT = "Question: Which arXiv CS sub-category does this paper belong to? Give 5 likely arXiv CS sub-categories as a comma-separated list ordered from most to least likely, in the form \"cs.XX\". Please think about the categorization in a step by step manner and avoid making false associations. Then provide your reasoning."

# Text
# PROMPT = "Question: Which arXiv CS sub-category does this paper belong to? Give 5 likely arXiv CS sub-categories as a comma-separated list ordered from most to least likely, in the form \"cs.XX\". Focus only on content in the actual text and avoid making false associations. Then provide your reasoning."

PROMPT = "Question: Which arXiv CS sub-category does this paper belong to? Give 5 likely arXiv CS sub-categories as a comma-separated list ordered from most to least likely, in the form \"cs.XX\", and provide your reasoning.\n\nAnswer: "

dataset = 'arxiv_2023'


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.0,
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_batch_size: int = 4,
    # max_gen_len: Optional[int] = None,
    max_gen_len: int = 2048,
    START: int = 0,
    END: int = 10000,
):

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    _, text = get_raw_text(use_text=True, seed=0)

    END = min(END, len(text))
    for nid in range(START, END):
        if not os.path.isfile(f'output/{dataset}/{nid}.json'):
            START = nid
            break

    print(f"START: {START}, END: {END}")
    prompts = [
        f"Abstract: {ab}\nTitle: {ti}\n{PROMPT}" for ti, ab in text]
    dialogs = [[{'role': 'user', 'content': p}] for p in prompts[START:END]]
    nids = np.arange(START, END)
    for i in tqdm(range(0, len(dialogs), max_batch_size)):
        batch_dialogs = dialogs[i:i+max_batch_size]
        batch_nids = nids[i:i+max_batch_size]
        try:
            results = generator.chat_completion(
                batch_dialogs,  # type: ignore
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )

            for nid, dialog, result in zip(batch_nids, batch_dialogs, results):
                result['prompt'] = dialog
                with open(f'output/{dataset}/{nid}.json', 'w')as f:
                    json.dump(result, f, indent=4)
        except:
            print("Error: ", batch_nids)


if __name__ == "__main__":
    fire.Fire(main)
