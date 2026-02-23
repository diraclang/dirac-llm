mlx_lm_lora.train \
--model ./mlx_model \
--train \
--train-mode sft \
--data dirac_data \
--batch-size 1 \
--learning-rate 1e-5 \
--iters 300
