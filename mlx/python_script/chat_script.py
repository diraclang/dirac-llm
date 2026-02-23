from mlx_lm import load, generate
from transformers import AutoTokenizer



# Path to your quantized model
model_path = "../llm_models/adapters"
model, _ = load(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Example chat messages
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "In diraclang, write a subroutine name it greeting, and inside do  Print 'Hello, World!', then call it"}
]

prompt = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)

def print_scratchpad_and_final(raw_output):
    if "</dirac>" in raw_output:
      raw_output = raw_output.split("</dirac>")[0] + "</dirac>"
      print(raw_output)


# Generate the response
response = generate(model, tokenizer, prompt=prompt)
print_scratchpad_and_final(response)
