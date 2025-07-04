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

    # Based on the usage example, the model expects the following prompt format.
    prompt = f"### Caption:{natural_text}\n### Tags:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("\nGenerating tags...")
    # Generate the output from the model using parameters from the example code
    output = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.8,
        top_p=0.95,
        repetition_penalty=1.1,
        top_k=40,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # The generated text includes the original prompt. We need to extract just the response.
    # The response is what comes after "### Tags:".
    parts = generated_text.split("### Tags:")
    if len(parts) > 1:
        tag_output = parts[1].strip()
    else:
        # Fallback if the output format is unexpected
        tag_output = "Could not find model response in generated text."


    print("\nGenerated Tags:")
    print(tag_output)

if __name__ == "__main__":
    main()
