#!/usr/bin/env python3

# BSD 3-Clause License
#
# Copyright (c) 2023
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import glob
import json
import logging
import sys

import torch
import tqdm
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

LLM_CHOICES = [
    "tiiuae/falcon-7b-instruct", "cerebras/Cerebras-GPT-1.3B",
    "tiiuae/falcon-40b-instruct", "mosaicml/mpt-7b-instruct"
]

OTHER_CHOICES = []


def gentext_to_line(generated_text):
    split = generated_text.split("\n")
    while not split[0].startswith("Summary: "):
        split = split[1:]
    split[0] = split[0].replace("Summary: ", "")
    return " ".join(split)


if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", type=str, help="Where to input the text labels from")
    parser.add_argument("output_json", type=str, help="Where to output the re-written text labels to")
    parser.add_argument("--model", type=str, default="tiiuae/falcon-7b-instruct", choices=LLM_CHOICES + OTHER_CHOICES, help="The LLM to use to summarize/rewrite the text labels")
    args = parser.parse_args()
    # yapf: enable

    # yapf: disable
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)-15s %(message)s")
    log = logging.getLogger()
    # yapf: enable

    with open(args.input_json, "r") as f:
        lines = json.load(f)

    model = args.model
    log.info(f"Loading {model}...")
    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    log.info("... done loading")

    log.info(f"Rewriting labels ...")
    new_lines = []
    lines = list(filter(lambda s: "tags" in s, lines))  # XXX
    for line in tqdm.tqdm(lines[:20]):
        # if "tags" not in line:
        #     new_lines.append(line)
        #     continue
        prompt = f"""
You are an expert at summarizing geographic data.  For each label, you report its percentage of the visible area and only one other salient fact.

Please summarize the following geographic data: {line}

Summary: The scene contains"""

        sequences = pipeline(
            prompt,
            # max_length=1024,
            # # max_length=1536,
            # # max_length=2048,
            # do_sample=True,
            # top_k=3,
            max_new_tokens=128,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            # temperature=0.,
        )
        new_line = gentext_to_line(sequences[0].get("generated_text"))
        new_lines.append(new_line)

    log.info(f"Writing rewritten labels to {args.output_json}")
    with open(args.output_json, "w") as f:
        json.dump(new_lines, f, indent=4)
