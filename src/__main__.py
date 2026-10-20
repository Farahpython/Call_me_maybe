import sys
import json
from pydantic import BaseModel, ValidationError
from llm_sdk import Small_LLM_Model


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, dict[str, str]]
    returns: dict[str, str]


class Prompt(BaseModel):
    prompt: str


def load_function_defs(file: str) -> list[FunctionDefinition]:
    try:
        with open(file, 'r') as f:
            d = json.load(f)
        return [FunctionDefinition(**item) for item in d]
    except FileNotFoundError:
        print(f"{f} not found...")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print("Invalid JSON syntax: ", e)
        sys.exit(1)
    except ValidationError as e:
        print("Invalid value,", e.errors()[0]["msg"])
        sys.exit(1)


def load_prompt(file: str) -> list[Prompt]:
    try:
        with open(file, 'r') as f:
            d = json.load(f)
        return [Prompt(**item) for item in d]
    except FileNotFoundError:
        print("File not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print("Invalid JSON syntax: ", e)
        sys.exit(1)
    except ValidationError as e:
        print("Invalid value,", e.errors()[0]["msg"])
        sys.exit(1)


functions = load_function_defs("data/input/functions_definition.json")
prompts = load_prompt("data/input/function_calling_tests.json")
model = Small_LLM_Model()


def request_builder(prompt: Prompt, defs: list[FunctionDefinition]) -> str:
    model_prompt = ""
    model_prompt += "Available Functions:\n"
    for d in defs:
        model_prompt += f"- {d.name}: {d.description}\n"
    model_prompt += f"User request: {prompt.prompt}\n"
    model_prompt += "Function to call:"
    return model_prompt


#def response_function(input_text: str) -> str:
    
    
