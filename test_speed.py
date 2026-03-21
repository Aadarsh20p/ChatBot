import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

print("=== OpenChat Speed Test ===")
print(f"PyTorch version: {torch.__version__}")
print(f"Available RAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB" if torch.cuda.is_available() else "CPU mode")

# Load model
openchat_path = os.path.join(os.getcwd(), "models", "openchat_3.5")
print(f"\nLoading model from: {openchat_path}")

start_time = time.time()
tokenizer = AutoTokenizer.from_pretrained(openchat_path, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    openchat_path,
    torch_dtype=torch.float16,  # Half precision
    device_map="cpu",
    local_files_only=True,
    low_cpu_mem_usage=True
)
model.eval()

load_time = time.time() - start_time
print(f"Model loaded in: {load_time:.1f} seconds")

# Test queries
test_queries = [
    "Hi",
    "What is Python?",
    "How do I write a for loop in Python?",
]

print("\n=== Testing Generation Speed ===")
for query in test_queries:
    print(f"\nQuery: {query}")
    
    # Prepare input
    prompt = f"GPT4 Correct User: {query}<|end_of_turn|>GPT4 Correct Assistant:"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)
    
    # Generate
    start_time = time.time()
    with torch.inference_mode():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=30,  # Short for testing
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    
    generation_time = time.time() - start_time
    
    # Decode
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    print(f"Response: {response}")
    print(f"Generation time: {generation_time:.2f} seconds")
    print(f"Tokens/second: {30/generation_time:.1f}")