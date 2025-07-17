import requests
from logger_config import get_logger

logger = get_logger(__name__)

# Arquivo faz a mesma coisa que o outro, mas usando Request. Só para comparar


class LMStudioClient:
    def __init__(self, base_url: str = "http://localhost:1234/v1"):
        self.base_url = base_url
        self.model = "dolphin3.0-llama3.1-8b"

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        url = f"{self.base_url}/chat/completions"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 1000,
        }
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"❌ Erro ao gerar resposta via LM Studio: {e}")
            return "❌ Desculpe, ocorreu um erro ao processar sua pergunta."
