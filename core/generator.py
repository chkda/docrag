from torch.fx.experimental.symbolic_shapes import lru_cache

from core.llm import SmolLLM

from typing import Any, Dict, List

class Generator:

    def __init__(self, device:str="cpu"):
        self.llm = SmolLLM(device=device)

    def build_prompt(
            self,
            query: str,
            chunks: List[Dict[str, Any]],
            car_model: str
    ) -> str:
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i} - Page {chunk['page_number']}"
                f"{chunk['text']}\n"
            )

        context = "\n".join(context_parts)

        prompt = f"""You are a helpful automotive assistant answering questions about car manuals. You must answer based ONLY on the provided context.

Car Model: {car_model}

Context from manual:
{context}

User Question: {query}

Instructions:
1. Answer the question based ONLY on the information in the context above
2. If the answer is not in the context, say "I couldn't find this information in the {car_model} manual"
3. Give a short answer within 2-3 lines

Answer:"""

        return prompt

    def generate_answer(
            self,
            query: str,
            chunks: List[Dict[str, Any]],
            car_model: str,
            max_tokens: int = 512,
    ) -> str:

        if not chunks:
            return f"I couldn't find any relevant information in the {car_model} manual."

        prompt = self.build_prompt(query, chunks, car_model)
        answer = self.llm.generate(
            prompt=prompt,
            max_tokens=max_tokens
        )

        return answer


def get_generator(device):
    return Generator(device=device)