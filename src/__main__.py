import sys
import json
import re
from pydantic import BaseModel, ValidationError, create_model
from llm_sdk import Small_LLM_Model
from .prompt_builders import prompt_builder, second_prompt_builder
from .classes import FunctionDefinition, Prompt


type_mapping = {
    "number": float,
    "string": str, 
    "integer": int,
    "boolean": bool
}


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


def response_function(input_text: str,
                      model: Small_LLM_Model, valid_funcs: list[str]) -> str:
    """
        This function receives the input text created by the prompt_builder
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


def response_arguements(input_text: str, model: Small_LLM_Model,
                         function: FunctionDefinition) -> dict:
    argument_so_far = ""
    input_ids = model.encode(input_text).flatten().tolist()
    while not argument_so_far.endswith("}"):
        scores_list = model.get_logits_from_input_ids(input_ids)
        scores = len(scores_list)
        allowed = []
        for i in range(scores):
            text = model.decode([i])
            if not text:
                continue
            possible_token = argument_so_far + text
            for name, p_type in function.parameters.items():
                python_type = type_mapping[p_type["type"]]
                if python_type == #the one in the schema (?)

        

def create_dynamic_model(f: FunctionDefinition) -> Type[BaseModel]:
    """
        This function creates and returns a pydantic class out of each
        function definition dynamically based on its name and given parameters.
    """
    fields = {}
    for name, p_type in f.parameters.items():
        python_type = type_mapping[p_type["type"]]
        fields[name] = (python_type, ...)
    ClassCreated = create_model(f.name, **fields)
    return ClassCreated


def main() -> None:
    functions = load_function_defs("data/input/functions_definition.json")
    prompts = load_prompt("data/input/function_calling_tests.json")
    model = Small_LLM_Model()
    valid_defs = []
    for f in functions:
        valid_defs.append(f.name)
    for prompt in prompts:
        input_prompt = prompt_builder(prompt, functions)
        chosen_function = response_function(input_prompt, model, valid_defs)
    """
        for f in functions:
            if f.name == chosen_function:
                second_input_prompt = second_prompt_builder(prompt, f)
                chosen_arguments = response_arguments(second_input_prompt, model, f)
                ClassCreated = create_dynamic_model(f)
                try:
                    validated_arguements = ClassCreated(**chosen_arguments)
                except ValidationError as e:
                    print("Invalid arguements: ", e)
    """

if __name__ == "__main__":
    main()
