output_prefix=Results_Perspective
python -m code.perspective_api \
--input_file results/Samples_for_Evaluation_50_chatgpt.xlsx \
--output_file results/Samples_for_Evaluation_50_chatgpt_${output_prefix}.jsonl

python -m code.perspective_api \
--input_file results/Samples_for_Evaluation_50_gpt-4.xlsx \
--output_file results/Samples_for_Evaluation_50_gpt-4_${output_prefix}.jsonl

python -m code.perspective_api \
--input_file results/Samples_for_Evaluation_50_llama2_70b.xlsx \
--output_file results/Samples_for_Evaluation_50_llama2_70b_${output_prefix}.jsonl

python -m code.perspective_api \
--input_file results/Samples_for_Evaluation_50_t5_large.xlsx \
--output_file results/Samples_for_Evaluation_50_t5_large_${output_prefix}.jsonl