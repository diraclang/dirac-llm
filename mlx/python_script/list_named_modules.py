from mlx_lm import load


# Update this path to your model directory
model_path = "./llm_models/model_complete"  # Path to fused model output
model, tokenizer = load(model_path)


print("Show model quantization size:")
print(model.layers[0].self_attn.q_proj.weight.dtype)  # Example to check quantization dtype 
print("List of named modules in the model:")
for name, module in model.named_modules():
    print(name)
