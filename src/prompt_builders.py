from .classes import Prompt, FunctionDefinition


def prompt_builder(prompt: Prompt, defs: list[FunctionDefinition]) -> str:
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


def second_prompt_builer(prompt: Prompt,
                         chosen_function: FunctionDefinition) -> str:
    model_prompt = ""
    model_prompt += f"User request: {prompt.prompt}\n"
    model_prompt += f"Selected function: {chosen_function.name}\n"
    model_prompt += "Required parameters:\n"
    for name, details in chosen_function.parameters.items():
        model_prompt += f"- {name}: {details['type']}\n"
        model_prompt += "Generate only a JSON object containing the arguments."
    return model_prompt
