from transformers import AutoModelForCausalLM, AutoTokenizer

class SmolLLM:
    def __init__(self, device:str = "cpu"):
        checkpoint = "HuggingFaceTB/SmolLM2-135M"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

    def generate(self, prompt:str, max_tokens:int):
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(inputs)
        return self.tokenizer.decode(outputs[0])
