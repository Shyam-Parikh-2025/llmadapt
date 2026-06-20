import os
import json
import inspect
import urllib.request
import urllib.error
from typing import get_type_hints, Callable


def _stringify_tool_output(out) -> str:
    """Serializes tool return values into valid JSON strings for model consumption."""
    if isinstance(out, str):
        return out
    try:
        return json.dumps(out)
    except TypeError:
        return str(out)


class Conversation:
    """Universal state engine managing decoupled chat histories."""
    def __init__(self, system_instruction: str = ""):
        self.history = []
        self.system_instruction = system_instruction

        if self.system_instruction:
            self.history.append({
                "role": "system", "content": self.system_instruction
            })

    def add_user_msg(self, text: str):
        self.history.append({"role": "user", "content": text})
    
    def add_model_msg(self, text: str = None, tool_calls: list = None, role: str = "assistant",
                       native: dict = None, native_provider: str = None):
        msg = {"role": role} 
        if text:
            msg["content"] = text
        if tool_calls:
            msg["tool_calls"] = tool_calls
        if native is not None:
            msg["_native"] = native
            msg["_native_provider"] = native_provider
        self.history.append(msg)
    
    def add_tool_response(self, function_name: str, output: str, tool_call_id: str = None):
        msg = {
            "role": "tool",
            "name": function_name,
            "content": output
        }
        if tool_call_id:
            msg["tool_call_id"] = tool_call_id
        self.history.append(msg)

    def export_for(self, provider: str, special_format: Callable[[list], list] = None) -> list:
        """Translates local flat memory frames into target external API layouts."""
        if special_format and callable(special_format):
            return special_format(self.history)

        if provider in ["ollama", "openai", "custom"]:
            return self.history
            
        elif provider == "anthropic":
            return self._export_anthropic()
            
        elif provider == "gemini":
            gemini_history = []
            for msg in self.history:
                if msg["role"] == "system": 
                    continue 
                if msg.get("_native_provider") == "gemini" and "_native" in msg:
                    gemini_history.append(msg["_native"])
                    continue
                if msg["role"] == "user":
                    gemini_history.append({"role": "user", "parts": [{"text": msg["content"]}]})
                elif msg["role"] == "assistant":
                    parts = []
                    if msg.get("content"):
                        parts.append({"text": msg["content"]})
                    if "tool_calls" in msg:
                        for tool_call in msg["tool_calls"]:
                            parts.append({
                                "functionCall": {
                                    "name": tool_call["function"]["name"],
                                    "args": tool_call["function"]["arguments"]
                                }
                            })
                    gemini_history.append({"role": "model", "parts": parts})
                elif msg["role"] == "tool":
                    gemini_history.append({
                        "role": "tool",
                        "parts": [{"functionResponse": {"name": msg["name"], "response": {"result": msg["content"]}}}]
                    })
            return gemini_history

    def _export_anthropic(self) -> list:
        """Structures conversation blocks to comply with Anthropic context constraints."""
        out = []
        pending_tool_results = []
        emitted_native_ids = set() 

        def flush():
            if pending_tool_results:
                out.append({"role": "user", "content": pending_tool_results.copy()})
                pending_tool_results.clear()

        for msg in self.history:
            if msg["role"] == "system":
                continue

            if msg.get("_native_provider") == "anthropic" and "_native" in msg:
                native_id = id(msg["_native"])
                if native_id in emitted_native_ids:
                    continue
                emitted_native_ids.add(native_id)
                flush()
                out.append(msg["_native"])
                continue

            if msg["role"] == "tool":
                pending_tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": msg["content"]
                })
                continue

            flush()

            if msg["role"] == "user":
                out.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                content = []
                if msg.get("content"):
                    content.append({"type": "text", "text": msg["content"]})
                for tool_call in msg.get("tool_calls", []):
                    content.append({
                        "type": "tool_use",
                        "id": tool_call["id"],
                        "name": tool_call["function"]["name"],
                        "input": tool_call["function"]["arguments"]
                    })
                out.append({"role": "assistant", "content": content if content else (msg.get("content") or "")})

        flush()
        return out


class ToolRegistry:
    """Introspective schema compiler mapping local Python logic to AI tools."""
    def __init__(self):
        self.functions_maps = {}
        self.schemas = {} 
    
    def register(self, python_function, schema: dict = None):
        if schema is None:
            func_name = python_function.__name__
            func_doc = inspect.getdoc(python_function) or "No description available."
            type_mapping = {
                str: "string", int: "integer", float: "number", bool: "boolean"
            }

            type_hints = get_type_hints(python_function)
            signature = inspect.signature(python_function)
            properties = {}
            required_params = []

            for param_name, param in signature.parameters.items():
                param_type = type_hints.get(param_name, str)
                gemini_type = type_mapping.get(param_type, "string")

                properties[param_name] = {
                    "type": gemini_type,
                    "description": f"The {param_name} parameter"
                }
                if param.default == inspect.Parameter.empty:
                    required_params.append(param_name)
            
            schema = {
                "name": func_name,
                "description": func_doc,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required_params
                }
            }
        else:
            func_name = schema["name"]

        self.functions_maps[func_name] = python_function
        self.schemas[func_name] = schema 
    
    def execute(self, name: str, args: dict):
        if name in self.functions_maps:
            return self.functions_maps[name](**args)
        raise ValueError(f"Tool {name} not found")

    def export_for(self, provider: str, special_format: Callable[[list], list] = None) -> list:
        if not self.schemas:
            return []

        schema_list = list(self.schemas.values()) 

        if special_format and callable(special_format):
            return special_format(schema_list)

        if provider in ["ollama", "openai"]:
            return [{"type": "function", "function": schema} for schema in schema_list]
        elif provider == "anthropic":
            return [{"name": s["name"], "description": s["description"], "input_schema": s["parameters"]} for s in schema_list]
        elif provider == "gemini":
            return [{"functionDeclarations": schema_list}]


class Agent:
    """Core runtime engine driving hot-swappable transport networks."""
    def __init__(self, provider: str, model: str, base_url: str = None, 
                 api_key: str = None, system_instruction: str = "", max_tool_iterations: int = 10):
        self.conversation = Conversation(system_instruction=system_instruction)
        self.tool_registry = ToolRegistry()
        self.switch_api(provider=provider, model=model, base_url=base_url, api_key=api_key)
        self.set_max_tool_iterations(max_tool_iterations) 

    def set_max_tool_iterations(self, n: int):
        """Sets the upper threshold for consecutive tool execution cycles."""
        if not isinstance(n, int) or n < 1:
            raise ValueError("max_tool_iterations must be an integer >= 1")
        self.max_tool_iterations = n

    def switch_api(self, provider: str, model: str, base_url: str = None, api_key: str = None):
        """Re-routes active transport endpoints and authorization vectors seamlessly."""
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key

        if base_url:
            self.url = base_url
        elif self.provider == "gemini":
            self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        elif self.provider == "openai":
            self.url = "https://api.openai.com/v1/chat/completions"
        elif self.provider == "anthropic":
            self.url = "https://api.anthropic.com/v1/messages"
        elif self.provider == "ollama":
            self.url = "http://localhost:11434/api/chat"
        else:
            self.url = ""

        if not self.api_key and self.provider != "ollama":
            self.api_key = os.getenv(f"{self.provider.upper()}_API_KEY")

    def add_tool(self, python_function: Callable, schema: dict = None):
        self.tool_registry.register(python_function, schema)

    def _send_request(self, payload: dict, headers: dict) -> dict:
        """Executes native standard library post requests using zero external code."""
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.url, data=data_bytes, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"\n[{self.provider.upper()} HTTP Error {e.code}]: {e.read().decode('utf-8')}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"\n[Network Unreachable]: Check route {self.url}. Reason: {e.reason}")

    def chat(self, user_input: str, custom_format_func: Callable[[list], list] = None) -> str:
        self.conversation.add_user_msg(user_input)

        iterations = 0 
        while True:
            iterations += 1
            if iterations > self.max_tool_iterations:
                raise RuntimeError(
                    f"Exceeded max_tool_iterations ({self.max_tool_iterations}) without a final "
                    f"text response. Call agent.set_max_tool_iterations(n) to raise this limit."
                )

            history = self.conversation.export_for(self.provider, special_format=custom_format_func)
            tools = self.tool_registry.export_for(self.provider, special_format=custom_format_func)
            headers = {"Content-Type": "application/json"}

            # --- GEMINI RUNTIME ---
            if self.provider == "gemini":
                if self.api_key: headers["x-goog-api-key"] = self.api_key
                payload = {"contents": history}
                if tools: payload["tools"] = tools
                if self.conversation.system_instruction:
                    payload["systemInstruction"] = {"parts": [{"text": self.conversation.system_instruction}]}
                
                res = self._send_request(payload, headers)
                parts = res['candidates'][0]['content']['parts']

                function_calls = [p['functionCall'] for p in parts if 'functionCall' in p]
                text_parts = [p['text'] for p in parts if 'text' in p]

                if function_calls:
                    tool_calls = [{"id": "", "function": {"name": fc['name'], "arguments": fc.get('args', {})}}
                                  for fc in function_calls]
                    self.conversation.add_model_msg(
                        text="".join(text_parts) or None,
                        tool_calls=tool_calls,
                        native={"role": "model", "parts": parts},
                        native_provider="gemini"
                    )
                    for fc in function_calls:
                        name, args = fc['name'], fc.get('args', {})
                        out = self.tool_registry.execute(name, args)
                        out_str = _stringify_tool_output(out) 
                        self.conversation.history.append({
                            "role": "tool", "name": name, "content": out_str,
                            "_native": {"role": "tool", "parts": [{"functionResponse": {"name": name, "response": {"result": out_str}}}]},
                            "_native_provider": "gemini"
                        })
                    continue
                text = "".join(text_parts)
                self.conversation.add_model_msg(text=text)
                return text
            
            # --- ANTHROPIC RUNTIME ---
            elif self.provider == "anthropic":
                headers.update({"x-api-key": self.api_key, "anthropic-version": "2023-06-01"})
                payload = {"model": self.model, "messages": history, "max_tokens": 4096}
                if tools: payload["tools"] = tools
                if self.conversation.system_instruction: payload["system"] = self.conversation.system_instruction

                res = self._send_request(payload, headers)
                t_calls, final_text = [], ""
                
                for block in res.get("content", []):
                    if block["type"] == "text": final_text += block["text"]
                    elif block["type"] == "tool_use":
                        t_calls.append({"id": block["id"], "function": {"name": block["name"], "arguments": block["input"]}})
                
                if t_calls:
                    self.conversation.add_model_msg(
                        text=final_text if final_text else None,
                        tool_calls=t_calls,
                        native={"role": "assistant", "content": res.get("content", [])},
                        native_provider="anthropic"
                    )
                    tool_result_blocks = []
                    for tc in t_calls:
                        out = self.tool_registry.execute(tc["function"]["name"], tc["function"]["arguments"])
                        out_str = _stringify_tool_output(out) 
                        tool_result_blocks.append({"type": "tool_result", "tool_use_id": tc["id"], "content": out_str})
                    native_user_msg = {"role": "user", "content": tool_result_blocks}
                    for tc, block in zip(t_calls, tool_result_blocks):
                        self.conversation.history.append({
                            "role": "tool", "name": tc["function"]["name"], "content": block["content"],
                            "tool_call_id": tc["id"],
                            "_native": native_user_msg, "_native_provider": "anthropic"
                        })
                    continue
                self.conversation.add_model_msg(text=final_text)
                return final_text

            # --- OPENAI / CUSTOM MOCK RUNTIME ---
            elif self.provider in ["openai", "custom"]:
                if self.api_key: headers["Authorization"] = f"Bearer {self.api_key}"
                payload = {"model": self.model, "messages": history}
                if tools: payload["tools"] = tools

                res = self._send_request(payload, headers)
                msg = res['choices'][0]['message']
                
                if msg.get("tool_calls"):
                    self.conversation.add_model_msg(text=msg.get("content"), tool_calls=msg["tool_calls"])
                    for tool_call in msg["tool_calls"]:
                        name = tool_call["function"]["name"]
                        args = json.loads(tool_call["function"]["arguments"]) if isinstance(tool_call["function"]["arguments"], str) else tool_call["function"]["arguments"]
                        out = self.tool_registry.execute(name, args)
                        self.conversation.add_tool_response(name, _stringify_tool_output(out), tool_call["id"]) 
                    continue
                text = msg.get("content", "")
                self.conversation.add_model_msg(text=text)
                return text
            
            # --- OLLAMA RUNTIME ---
            elif self.provider == "ollama":
                payload = {"model": self.model, "messages": history, "stream": False}
                if tools: payload["tools"] = tools
                    
                res = self._send_request(payload, headers)
                msg = res.get("message", {})
                
                if msg.get("tool_calls"):
                    self.conversation.add_model_msg(tool_calls=msg["tool_calls"])
                    for tool_call in msg["tool_calls"]:
                        name, args = tool_call["function"]["name"], tool_call["function"]["arguments"]
                        out = self.tool_registry.execute(name, args)
                        self.conversation.add_tool_response(name, _stringify_tool_output(out)) 
                    continue
                text = msg.get("content", "")
                self.conversation.add_model_msg(text=text)
                return text
