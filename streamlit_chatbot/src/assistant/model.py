from typing import Any, List, Optional, Literal
from langchain_core.language_models.llms import LLM
from pydantic import BaseModel
import requests


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "function"]
    content: Optional[str] = None


class LangchainAPIAdapter(LLM):
    '''
    Bridge for langchain to use LLM model over HTTP. 
    '''

    route: str
    llm_type: str = 'RemoteLLM'

    @property
    def _llm_type(self) -> str:
        return self.llm_type

    def _call(
        self,
        prompt: str,
        role: Literal['user', 'assistant', 'system'] = 'user',
        temperature: float = 0.7,
        max_new_tokens: int = 1000,
        history: Optional[List[ChatMessage]] = None,
        **kwargs: Any,
    ) -> str:
        # 請思考這段
        payload = {
            "role": role,
            "content": prompt,
            "history": history,
            "temperature": temperature,
            "max_new_tokens":  max_new_tokens
        }

        res = requests.post(self.route, json=payload).json()

        return res['content']


class Assistant:
    """
    Chatbot assistant
    """

    def __init__(self, url: str, system_prompt: str) -> None:
        self.url = url
        self.system_prompt = system_prompt
        self.model = LangchainAPIAdapter(
            route=self.url,
        )

    def chat(self, prompt: str, history: List[ChatMessage]) -> str:
        system_message = [{"role": "system", "content": self.system_prompt}]
        history = system_message + history
        response = self.model.generate(
            prompts=[prompt],
            temperature=0.7,
            max_new_tokens=4096,
            history=history)
        return response.generations[0][0].text
