from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Choose the model
model_id = "mistralai/Mistral-7B-Instruct-v0.1"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Set up pipeline
chat = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.1
)

# Prompt wrapper
def ask_bot(prompt):
    system_prompt = "<s>[INST] " + prompt + " [/INST]"
    output = chat(system_prompt)[0]["generated_text"]
    return output.replace(system_prompt, "").strip()


if __name__ == "__main__":
    print("🤖 Chatbot (type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = ask_bot(user_input)
        print("Bot:", response)
