import pandas as pd

# fname = "dataset/english/un_dataset/data_ref_llama2_70b.csv"
# output_file = "dataset/english/un_dataset/data_ref_llama2_70b_corrected.csv"

fname = "dataset/english/BABE/evaluation/Samples_for_Evaluation_100_llama2_70b.csv"
output_file = "dataset/english/BABE/evaluation/Samples_for_Evaluation_100_llama2_70b_corrected.csv"

def extract_unbiased_sentence(text):
    texts = text.split("\n")
    for idx, _text in enumerate(texts):
        if "Unbiased Sentence:" in _text:
            return texts[idx+1]


data = pd.read_csv(fname, sep=",")

data.unbiased_text = data.unbiased_text.apply(lambda x: extract_unbiased_sentence(x))
data.to_csv(output_file, sep=",", index=False)
