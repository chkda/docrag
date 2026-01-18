from transformers import AutoModelForCausalLM, AutoTokenizer

class TransformersLLM:
    def __init__(self, device:str = "cpu"):
        checkpoint = "HuggingFaceTB/SmolLM2-135M-Instruct"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

    def generate(self, prompt:str, max_tokens:int):
        messages = [{"role": "user", "content": prompt}]
        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(inputs, max_new_tokens=max_tokens)
        output_tokens = outputs[0][inputs.shape[1]:]
        return self.tokenizer.decode(output_tokens, skip_special_tokens=True)
