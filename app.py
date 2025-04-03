import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load CodeLlama Model
MODEL_NAME = "codellama/CodeLlama-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float32, device_map="cpu"  # Use CPU instead of CUDA
)


def detect_and_fix_bug(code_snippet):
    """
    Detects and fixes bugs in a given code snippet using CodeLlama.
    Also identifies the exact line of error.
    """
    prompt = f"""
### Buggy Code:
{code_snippet}

### Task:
1. Identify if there is an error in the code.
2. Highlight the exact line where the error occurs.
3. Provide a corrected version of the code.

### Analysis:
"""
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

    # Generate output with proper settings
    output = model.generate(
        **inputs,
        max_length=512,
        temperature=0.2,
        do_sample=True,  # Enable sampling for better variety
        pad_token_id=tokenizer.eos_token_id  # Suppress warning
    )

    # Decode output
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response


# Example Buggy Code
buggy_code = """
def divide_numbers(a, b):
    return a / b   # Possible division by zero error

print(divide_numbers(10, 0))
"""

# Get the analysis
result = detect_and_fix_bug(buggy_code)
print("Code Analysis and Fix:\n", result)

# Save Model & Tokenizer Locally
model.save_pretrained("./saved_codellema")
tokenizer.save_pretrained("./saved_codellema")
print("Model saved successfully!")

# Load Saved Model from Local Storage
MODEL_PATH = "./saved_codellema"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

print("Model loaded successfully!")