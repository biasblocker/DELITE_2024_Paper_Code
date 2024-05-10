import os
from argparse import ArgumentParser

import langchain
import pandas as pd
from code.prompt_template import standard_prompt
from dotenv import load_dotenv
from langchain import PromptTemplate, LLMChain
from langchain.cache import SQLiteCache
from langchain.llms import OpenAI
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('--input_file')
parser.add_argument('--output_file')
parser.add_argument('--model')

args = parser.parse_args()

model = args.model
input_file = args.input_file
output_file = args.output_file

langchain.llm_cache = SQLiteCache(database_path=".benchmarking_experiments.db")


def correct_bias(sentence, llm_chain):
    translation = llm_chain.run(sentence=sentence)
    return translation


template = standard_prompt

if ".csv" in input_file:
    dataset = pd.read_csv(input_file, sep=",")
    biased_sentences = list(set(dataset["biased_text"].tolist()))
elif ".tsv" in input_file:
    dataset = pd.read_csv(input_file, sep="\t")
    biased_sentences = list(set(dataset["biased_text"].tolist()))
elif ".txt" in input_file:
    biased_sentences = []
    with open(input_file, 'r', encoding='UTF-8') as file:
        while line := file.readline():
            biased_sentences.append(line.rstrip())

prompt = PromptTemplate(
    input_variables=["sentence"],
    template=template)

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")

llm = OpenAI(model_name=model,
             openai_api_key=os.environ["OPENAI_API_KEY"],
             temperature=0.7,
             request_timeout=250
             )

llm_chain = LLMChain(llm=llm, prompt=prompt, output_key='translation')

outputs = []
for sample in tqdm(biased_sentences, total=len(biased_sentences)):
    if isinstance(sample, float):
        continue

    outputs.append({
        'biased_text': sample,
        'unbiased_text': correct_bias(sentence=sample, llm_chain=llm_chain)})

if "txt" in output_file:
    with open(output_file, 'w') as f:
        for line in outputs:
            if len(line["biased_text"].strip()) == 0:
                continue

            f.write("====Original Sentence====")
            f.write('\n')
            f.write('\n')
            f.write(line["biased_text"])
            f.write('\n')
            f.write('\n')

            f.write("====Corrected Sentence====")
            f.write('\n')
            f.write('\n')
            f.write(line["unbiased_text"])
            f.write('\n')
            f.write('\n')

else:
    outputs = pd.DataFrame(outputs)
    outputs.to_csv(output_file, index=False, sep=',')
