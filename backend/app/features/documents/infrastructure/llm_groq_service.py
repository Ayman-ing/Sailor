from groq import Groq
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class LLMGroqService:
    def __init__(self, client: Groq):
        self.client = client

    def summarize_table(self, table_content: str) -> str:
        """
        Summarizes table content using a smaller, faster model.
        """
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a master at summarizing tables. "
                            "Your task is to provide a concise, natural language summary of the given table content. "
                            "Focus on the key information and main trends presented in the table. "
                            "The summary should be easily understandable and suitable for embedding and retrieval in a RAG system."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Please summarize the following table:\n\n{table_content}",
                    },
                ],
                model=settings.groq_model,
                temperature=0.2,
                max_tokens=256,
                top_p=1,
                stop=None,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            # In case of API error, return the original content
            logger.error(f"Error summarizing table: {e}", exc_info=True)
            return table_content

    def summarize_code(self, code_content: str) -> str:
        """
        Summarizes a code snippet using a model adept at code analysis.
        """
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert programmer. Your task is to provide a high-level, natural language summary of the given code snippet. "
                            "Explain the code's purpose, its main functionality, and what it accomplishes. "
                            "Do not describe the code line-by-line. The summary should be suitable for embedding and retrieval in a RAG system."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Please summarize the following code snippet:\n\n```\n{code_content}\n```",
                    },
                ],
                model=settings.groq_model,
                temperature=0.2,
                max_tokens=512,
                top_p=1,
                stop=None,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            # In case of API error, return the original content
            logger.error(f"Error summarizing code: {e}", exc_info=True)
            return code_content

def get_llm_service() -> LLMGroqService:
    groq_client = Groq(api_key=settings.groq_api_key)
    return LLMGroqService(client=groq_client)
