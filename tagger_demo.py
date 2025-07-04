import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    """
    This demo script uses the John6666/llama-tagger-HF-GPTQ-4bits model
    to convert a natural language description into a list of tags.
    """
    model_id = "John6666/llama-tagger-HF-GPTQ-4bits"
    
    print(f"Loading model: {model_id}")
    # The `device_map="auto"` will place the model on the available GPU.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("Model and tokenizer loaded successfully.")

    natural_text = (
        "A beautiful illustration of Aqua from the anime series Konosuba. "
        "She is depicted dancing gracefully and joyfully in the middle of a heavy downpour at night on a city street. "
        "Rain is splashing all around her, catching the glow from the streetlights. "
        "She has a radiant smile on her face, with her long blue hair and signature outfit soaked with rain. "
        "The overall mood is cheerful and magical."
    )

    print("\nNatural text description:")
    print(natural_text)

    # The model uses a specific chat template for tag generation.
    messages = [
        {
            "role": "user",
            "content": f"Generate tags for the following text. Keep it brief, clear and focused.\n\n{natural_text}"
        }
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("\nGenerating tags...")
    # Generate the output from the model
    output = model.generate(
        **inputs,
        max_new_tokens=256,
    )
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # The generated text includes the original prompt, so we need to extract just the response.
    # The response is what comes after [/INST]
    response_start_index = generated_text.find("[/INST]")
    if response_start_index != -1:
        tag_output = generated_text[response_start_index + len("[/INST]"):].strip()
    else:
        tag_output = "Could not find model response in generated text."


    print("\nGenerated Tags:")
    print(tag_output)

if __name__ == "__main__":
    main()
