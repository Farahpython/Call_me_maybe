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
    """
        This function loads the function_definition.json file
        and returns a list of FunctionDefinition
    """
    try:
        with open(file, 'r') as f:
            d = json.load(f)
        return [FunctionDefinition(**item) for item in d]
    except FileNotFoundError:
        print(f"{file} not found...")
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
        print(f"{file} not found")
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
    """
        This function takes the loaded function definitions and a single prompt
        and returns them as a single text so its passed to the model.
    """
    model_prompt = ""
    model_prompt += "Available Functions:\n"
    for d in defs:
        model_prompt += f"- {d.name}: {d.description}\n"
    model_prompt += f"User request: {prompt.prompt}\n"
    model_prompt += "Function to call:"
    return model_prompt


def response_function(input_text: str, model: Small_LLM_Model,
    valid_funcs: list[str]) -> str:
    """
        This function receives the input text created by the request_builder
        function, the model, and a list of valid function names and returns
        the name of the chosen function.
    """
    vocab = []
    allowed = []
    chosen_function = ""
    ids_list = model.encode(input_text).flatten().tolist()  #returns token IDs
    vocab_scores = model.get_logits_from_input_ids(ids_list)  #returns raw logits
    vocab_size = len(vocab_scores)
    for i in range(vocab_size):
        vocab += [model.decode([i])]
    for i in range(vocab_size):
        text = vocab[i]
        if not text:
            continue
        candidate = chosen_function + text
        if any(name.startswith(candidate) for name in valid_funcs):
            allowed.append(i)
    return chosen_function


if __name__ == "__main__":
    text = request_builder(prompts[0], functions)
    response_function(text, model, functions)
