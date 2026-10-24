import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}


def save_captions(captions, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + '.tmp')
    with temporary_path.open('w', encoding='utf-8') as output_file:
        json.dump(captions, output_file, ensure_ascii=False)
    os.replace(temporary_path, output_path)


def generate(dataset_root, model_path, output_path, image_dir_a, image_dir_b, prompt):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model = model.to(device).eval()
    config = model.config
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        model_max_length=config.tokenizer_model_max_length,
        padding_side=config.tokenizer_padding_side,
    )

    if os.path.isfile(output_path):
        with open(output_path, 'r', encoding='utf-8') as captions_file:
            captions = json.load(captions_file)
    else:
        captions = {}

    processed = 0
    for split in ('train', 'val', 'test'):
        for image_dir in (image_dir_a, image_dir_b):
            folder = Path(dataset_root) / split / image_dir
            for image_path in sorted(path for path in folder.iterdir()
                                     if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES):
                key = os.path.abspath(str(image_path))
                if key in captions:
                    continue
                output_text, _ = model.chat(prompt=prompt, image=key, tokenizer=tokenizer)
                captions[key] = output_text
                processed += 1
                if processed % 50 == 0:
                    save_captions(captions, output_path)
                    print(f'Generated {processed} new captions; total {len(captions)}')

    save_captions(captions, output_path)
    print(f'Saved {len(captions)} image captions to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate MMChange image descriptions with TinyLLaVA')
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--model-path', required=True, help='Local TinyLLaVA model directory')
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--image-dir-a', default='A')
    parser.add_argument('--image-dir-b', default='B')
    parser.add_argument('--prompt', default='What are these?')
    args = parser.parse_args()
    generate(args.dataset_root, args.model_path, args.output_json,
             args.image_dir_a, args.image_dir_b, args.prompt)
