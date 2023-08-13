# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional

import fire

from llama import Llama
import json
from tqdm import tqdm
import pandas as pd
import os

PROMPT = "Question: Which of the following category does this product belong to: 1) Home & Kitchen, 2) Health & Personal Care, 3) Beauty, 4) Sports & Outdoors, 5) Books, 6) Patio, Lawn & Garden, 7) Toys & Games, 8) CDs & Vinyl, 9) Cell Phones & Accessories, 10) Grocery & Gourmet Food, 11) Arts, Crafts & Sewing, 12) Clothing, Shoes & Jewelry, 13) Electronics, 14) Movies & TV, 15) Software, 16) Video Games, 17) Automotive, 18) Pet Supplies, 19) Office Products, 20) Industrial & Scientific, 21) Musical Instruments, 22) Tools & Home Improvement, 23) Magazine Subscriptions, 24) Baby Products, 25) NAN, 26) Appliances, 27) Kitchen & Dining, 28) Collectibles & Fine Art, 29) All Beauty, 30) Luxury Beauty, 31) Amazon Fashion, 32) Computers, 33) All Electronics, 34) Purchase Circles, 35) MP3 Players & Accessories, 36) Gift Cards, 37) Office & School Supplies, 38) Home Improvement, 39) Camera & Photo, 40) GPS & Navigation, 41) Digital Music, 42) Car Electronics, 43) Baby, 44) Kindle Store, 45) Kindle Apps, 46) Furniture & Decor? Give 5 likely categories as a new line ordered from most to least likely, and provide your reasoning."


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.0,
    top_p: float = 0.9,
    max_seq_len: int = 1024,
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

    dataset = 'ogbn-products'
    with open(f'dataset/{dataset}.csv') as f:
        df = pd.read_csv(f)
    df['title'].fillna("", inplace=True)
    df['content'].fillna("", inplace=True)

    for nid in range(START, END):
        if not os.path.isfile(f'output/{dataset}/{nid}.json'):
            START = nid
            break

    print(f"START: {START}, END: {END}")
    prompts = [f"Title: {ti[:50]}\nContent: {c[:150]}\n{PROMPT}" for ti,
               c in zip(df['title'], df['content'])]
    dialogs = [[{'role': 'user', 'content': p}] for p in prompts[START:END]]
    nids = df['nid'].values[START:END]
    for i in tqdm(range(0, len(dialogs), max_batch_size)):
        batch_dialogs = dialogs[i:i+max_batch_size]
        batch_nids = nids[i:i+max_batch_size]
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


if __name__ == "__main__":
    fire.Fire(main)
