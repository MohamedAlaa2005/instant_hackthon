import os
from typing import Generator, Optional, Type

from google import genai
from google.genai import types
from pydantic import BaseModel


class Gemini:
    """Wrapper for Google Gemini API."""

    def __init__(
        self,
        model: str = "gemini-3.5-flash-lite",
        api_key: Optional[str] = None,
        system_instruction: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is missing.")

        self.client = genai.Client(api_key=self.api_key)
        self.system_instruction = system_instruction

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
        max_output_tokens: Optional[int] = None,
    ) -> str:
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            system_instruction=system_instruction or self.system_instruction,
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )
        return response.text or ""

    def generate_stream(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction or self.system_instruction,
        )
        response = self.client.models.generate_content_stream(
            model=self.model,
            contents=prompt,
            config=config,
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text

    def generate_structured(
        self,
        prompt: str,
        response_schema: Type[BaseModel],
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
    ) -> BaseModel:
        config = types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=response_schema,
            system_instruction=system_instruction or self.system_instruction,
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )
        return response_schema.model_validate_json(response.text)
