# frontend: OpenWebUI Pipe (Python side) — Legal dataset
import json
import time
import httpx
from typing import Callable
from fastapi import Request
from pydantic import BaseModel


class Pipe:
    class Valves(BaseModel):
        pass

    def __init__(self):
        self.valves = self.Valves()

    def get_capabilities(self):
        return {
            "allow_file_upload": True,
            "allow_image_input": False,
            "allow_code_interpreter": False,
        }

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __request__: Request,
        __metadata__: dict,
        __event_emitter__: Callable,
    ):
        start = time.time()
        user_id = __metadata__.get("user_id", "default_user")
        chat_id = __metadata__.get("chat_id", "default_chat")
        chat_messages = [i for i in body["messages"] if i["role"] != "system"]
        files = [i["url"] for i in (__metadata__.get("files") or [])]

        # Skip internal OpenWebUI task prompts (follow-up suggestions, title generation, etc.)
        last_user = next((m for m in reversed(chat_messages) if m["role"] == "user"), None)
        if last_user and (last_user.get("content") or "").strip().startswith("### Task:"):
            return

        await __event_emitter__(
            {
                "type": "status",
                "data": {
                    "description": "Connecting to Pneuma Seeker...",
                    "done": False,
                    "hidden": False,
                },
            }
        )

        async with httpx.AsyncClient(timeout=httpx.Timeout(900.0)) as client:
            end_initialization = time.time()
            async with client.stream(
                "POST",
                "http://pneuma-seeker:8000/chat",
                json={
                    "messages": chat_messages,
                    "files": files,
                    "user_id": user_id,
                    "chat_id": chat_id,
                    "dataset": "legal",
                },
            ) as response:
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        message_data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    sender = message_data.get("sender")
                    text = message_data.get("text", "")

                    if sender == "log":
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {
                                    "description": (
                                        text[5:] if text.startswith("LOG: ") else text
                                    ),
                                    "done": False,
                                    "hidden": False,
                                },
                            }
                        )
                    elif sender == "assistant":
                        await __event_emitter__(
                            {"type": "chat:message:delta", "data": {"content": text.replace("~", "\\~")}}
                        )
                    elif sender == "done":
                        try:
                            sidebar_resp = await client.post(
                                f"http://pneuma-seeker:8000/combined/html/{user_id}/{chat_id}",
                                json=body,
                                timeout=30.0,
                            )
                            if sidebar_resp.status_code == 200:
                                launcher_html = sidebar_resp.text
                                launcher_html = launcher_html.replace("<body", '<body style="min-height:800px;"')
                                launcher_html = launcher_html.replace("```", "`\u200b``")
                                html_block = (
                                    "```html\n"
                                    "<!-- PNEUMA_STATE_START -->\n"
                                    f"{launcher_html}\n"
                                    "<!-- PNEUMA_STATE_END -->\n"
                                    "```"
                                )
                                await __event_emitter__(
                                    {"type": "chat:message:delta", "data": {"content": "\n\n" + html_block}}
                                )
                        except Exception:
                            pass

                        elapsed = end_initialization - start
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {
                                    "description": f"{text} (connection initialization: {elapsed:.2f} seconds)",
                                    "done": True,
                                    "hidden": False,
                                },
                            }
                        )
                        break
