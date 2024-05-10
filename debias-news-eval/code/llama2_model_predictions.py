import os
from argparse import ArgumentParser

import pandas as pd
import torch
from code.prompt_template import standard_prompt
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)

load_dotenv()
access_token_read = os.environ.get("HF_TOKEN")
login(token=access_token_read)

parser = ArgumentParser()

parser.add_argument('--input_file')
parser.add_argument('--output_file')
parser.add_argument('--model')
parser.add_argument('--device')

args = parser.parse_args()

model_name = args.model

os.environ["CUDA_VISIBLE_DEVICES"] = args.device

input_file = args.input_file
output_file = args.output_file

if ".csv" in input_file:
    dataset = pd.read_csv(input_file, sep=",")
else:
    dataset = pd.read_csv(input_file, sep="\t")

biased_sentences = list(set(dataset["biased_text"].tolist()))

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id  # for open-ended generation

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    use_auth_token=True
)
generation_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    trust_remote_code=True,
    device_map="auto",  # finds GPU
)

prompt = standard_prompt

outputs = []
for sample in tqdm(biased_sentences, total=len(biased_sentences)):
    if isinstance(sample, float):
        continue
    text = prompt.format(
        sentence=sample)

    sequences = generation_pipe(
        text,
        max_length=256,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=10,
        temperature=0.7,
        top_p=0.9
    )

    outputs.append({
        'biased_text': sample,
        'unbiased_text': sequences[0]["generated_text"]})

pd.DataFrame(outputs).to_csv(output_file, sep=",", index=False)
