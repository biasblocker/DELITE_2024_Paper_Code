import pandas as pd
import os
from argparse import ArgumentParser
from tqdm import tqdm
from peft import PeftModel, PeftConfig
from transformers import T5ForConditionalGeneration, AutoTokenizer


def prepare_input(sentence: str):
    input_ids = tokenizer(sentence, max_length=256, return_tensors="pt").input_ids
    return input_ids


def inference(sentence: str) -> str:
    input_data = prepare_input(sentence=sentence)
    input_data = input_data.to(model.device)
    outputs = model.generate(inputs=input_data, max_length=256)
    result = tokenizer.decode(token_ids=outputs[0], skip_special_tokens=True)
    return result


parser = ArgumentParser()
parser.add_argument('--input_file')
parser.add_argument('--output_file')
parser.add_argument('--peft_model_id')
parser.add_argument('--device')

args = parser.parse_args()

peft_model_id = args.peft_model_id

os.environ["CUDA_VISIBLE_DEVICES"] = args.device

input_file = args.input_file
output_file = args.output_file

if ".csv" in input_file:
    dataset = pd.read_csv(input_file, sep=",")
else:
    dataset = pd.read_csv(input_file, sep="\t")

biased_sentences = list(set(dataset["biased_text"].tolist()))

config = PeftConfig.from_pretrained(peft_model_id)

model = T5ForConditionalGeneration.from_pretrained(config.base_model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

model = PeftModel.from_pretrained(model, peft_model_id)
model.eval()

outputs = []
for sample in tqdm(biased_sentences, total=len(biased_sentences)):
    if isinstance(sample, float):
        continue
    outputs.append({
        'biased_text': sample,
        'unbiased_text': inference(f"debias: {sample} </s>")})

pd.DataFrame(outputs).to_csv(output_file, sep=",", index=False)
