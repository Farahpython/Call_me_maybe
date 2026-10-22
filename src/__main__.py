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


def response_function(input_text: str,
                      model: Small_LLM_Model, valid_funcs: list[str]) -> str:
    """
        This function receives the input text created by the request_builder
        function, the model, and a list of valid function names and returns
        the name of the chosen function.
    """
    token_so_far = ""  # generated so far
    # from tokens to token IDs
    input_ids = model.encode(input_text).flatten().tolist()
    # goal is to inspect the vocabulary, not the prompt.
    while token_so_far not in valid_funcs:
        """ check how many possible tokens exist.
        retrive list of scores """
        scores_list = model.get_logits_from_input_ids(input_ids)
        scores = len(scores_list)  # retrieves number of logits
        allowed = []
        for i in range(scores):
            text = model.decode([i])  # next token we're testing
            if not text:  # prevent empty strings
                continue
            possible_token = token_so_far + text
            if any(name.startswith(possible_token) for name in valid_funcs):
                allowed.append(i)
        best_token = max(allowed, key=lambda i: scores_list[i])
        if not allowed:
            return token_so_far
        token_so_far += model.decode([best_token])
        input_ids.append(best_token)
    return token_so_far


if __name__ == "__main__":
    text = request_builder(prompts[10], functions)
    valid_funcs = [f.name for f in functions]
    result = response_function(text, model, valid_funcs)
    print(result)
