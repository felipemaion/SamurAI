import lmstudio as lms
from logger_config import get_logger

logger = get_logger(__name__)


class LMStudioClient:
    def __init__(self, model_name: str = "dolphin3.0-llama3.1-8b"):
        try:
            self.model = lms.llm(model_name)
            logger.info(f"✅ Modelo LMStudio carregado: {model_name}")
        except Exception as e:
            logger.error(
                f"❌ Erro ao inicializar LMStudio com o modelo {model_name}: {e}"
            )
            self.model = None

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        if not self.model:
            logger.error("⚠️ Modelo LMStudio não inicializado.")
            return "⚠️ Desculpe, o modelo não está disponível."

        try:
            full_prompt = prompt
            if system_prompt:
                # Se desejar combinar os prompts, por exemplo:
                full_prompt = f"{system_prompt.strip()}\n\n{prompt.strip()}"

            result = self.model.respond(full_prompt)
            return str(result).strip()  # TODO: ESTRUTURAR A RESPOSTA COMO JSON
        # ver: https://lmstudio.ai/docs/python/llm-prediction/structured-response
        except Exception as e:
            logger.error(f"❌ Erro ao gerar resposta via LM Studio: {e}")
            return "❌ Desculpe, ocorreu um erro ao processar sua pergunta."
