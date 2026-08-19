"""
Lightning AI (LitAI) chat client.

Same models as the Gemini API but routed through Lightning's OpenAI-compatible
endpoint, which is not subject to the Google free tier's 15 requests/minute
cap. That cap was aborting benchmark runs part-way through.

Only an API key is needed. The `litai` SDK also wants a teamspace resolved
through a platform login, which the `sk-lit-` model key alone cannot do - the
HTTP endpoint sidesteps that entirely, so there is no SDK dependency here.

Usage:  python -m src.generation.lightning_llm "your prompt"
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE_URL = "https://lightning.ai/api/v1"
DEFAULT_MODEL = "google/gemini-3.1-flash-lite-preview"
MAX_RETRIES = 4
TIMEOUT = 90


class LightningLLM:
    """Minimal chat client over Lightning's OpenAI-compatible API."""

    def __init__(self, model=DEFAULT_MODEL, api_key=None, base_url=BASE_URL):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("LIGHTNING_API_KEY")
        if not self.api_key:
            raise ValueError("LIGHTNING_API_KEY is not set - add it to .env")

    def _post(self, payload):
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        last = None
        for attempt in range(MAX_RETRIES):
            try:
                with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                    return json.loads(response.read())
            except urllib.error.HTTPError as exc:
                last = exc
                # 429/5xx are worth another try; a 400 never is.
                if exc.code not in (429, 500, 502, 503, 504):
                    raise
            except Exception as exc:
                last = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(2**attempt)
        raise last

    def chat(self, prompt, system_instruction=None, temperature=0.7, json_mode=False):
        """Send one prompt, return the assistant's text."""
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        return self._post(payload)["choices"][0]["message"]["content"]

    # Gemini-compatible aliases so callers can swap providers without edits.
    def generate(self, prompt, system_instruction=None, temperature=0.7, **_):
        return self.chat(prompt, system_instruction, temperature)

    def generate_stream(self, prompt, system_instruction=None, temperature=0.7, **_):
        """No server-side streaming here; yields the answer as one chunk."""
        yield self.chat(prompt, system_instruction, temperature)


def main():
    from dotenv import load_dotenv
    from pathlib import Path

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Name the three most common causes of cirrhosis."
    print(LightningLLM().chat(prompt))


if __name__ == "__main__":
    main()
