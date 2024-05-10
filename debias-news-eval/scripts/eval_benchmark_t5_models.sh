python -m app.eval_benchmark.t5_model_predictions \
--peft_model_id models/flan-t5-large_debiaser_debias_1e-3_wnc \
--device "1" \
--input_file --- \
--output_file ---

python -m app.eval_benchmark.t5_model_predictions \
--peft_model_id models/t5_large_debiaser_debias_1e-3_wnc \
--device "1" \
--input_file --- \
--output_file ---

