*This activity has been created as part of the 42 curriculum by fadarwis.*
*If you misuse my code i will haunt you :)*

# Description

**Call_me_maybe** is a project that introduces function calling in Large Language Models by creating a function calling tool that translates natural language prompts into structured function calls.

**Function calling**: models can interact with external systems and access data outside their training data.

This project also introduces **constrained" decoding**, which is filtering the model's vocabulary at each generation step to create structured output. Even though it guarantees the shape of the answer, it does not guarantee the content of it.

Before the model samples each token, the system checks which tokens would keep the output valid according to the json schema and it masks out everything else.

This project uses the **Qwen/Qwen3-0.6** model, which is the latest generation of LLMs in the Qwen family. It is a text-generation model thatis trained on 3.6 trillion tokens across 119 languages.

# Instructions

## To set up the environment 

```
make install
```
## To launch the program

```
make run 
```
## To launch the program using python's debugger

```
make debug
```
## To remove generated python files and the virtual environment

```
make clean
```
## To activate the virtual environment 

```
make activate
```
## For flake8 and mypy checks

```
make lint
```

# Resources

1. https://azure.microsoft.com/en-us/resources/cloud-computing-dictionary/what-are-large-language-models-llms
2. https://www.sanity.io/glossary/constrained-decoding
3. https://developers.openai.com/api/docs/guides/function-calling?api-mode=responses
4. https://nanonets.com/cookbooks/structured-llm-outputs/basic-concepts/sampling
5. https://huggingface.co/spaces/hesamation/primer-llm-embedding?section=what_are_embeddings?
6. https://huggingface.co/learn/llm-course/chapter2/4
7. Youtube:
  - Transformers, the tech behind LLMs | Deep Learning Chapter 5 - 3Blue1Brown
   - https://www.youtube.com/watch?v=Ibno_G3tCZM
8. https://machinelearningmastery.com/the-complete-guide-to-using-pydantic-for-validating-llm-outputs/

## Algorithm explanation

1. A list of validated functions (loaded from the given input file) is created.
2. For each prompt, an **input_prompt** is generated based on the given prompt and the list of
   validated functions. The **prompt_builder** function will return a prompt text that will list all the available functions (names and descriptions) and the prompt itself.
3. **response_function**: This function returns the chosen function name based on the **input_prompt** passed using constrained decoding to restrict the model's selection to valid function names.
4. **response_arguments**: This function returns the validated arguments of the chosen function using different mechanisms: few-shot prompting, providing the model with the json skeleton, candidate extraction for numeric arguments, and FSM (Finite State Machine) for string arguments.
5. **create_dynamic_model**: create a dynamic Pydantic model for each function definiton to validate generated arguments. 

## Design decision

The use of constrained decoding enhances the accuracy and reliability of generated outputs. By implementing constraints, you can tailor model behavior to meet the specific needs of your application.

In this project, constrained decoding was chosen to ensure that generated function names are valid and that arguments follow the required JSON syntax. Different approaches were implemented for handling parameter types, including finite-state machine (FSM) logic for JSON string generation and constrained token selection for numerical and Boolean values.

Additionally, Pydantic was used to dynamically create models based on function definitions, allowing generated arguments to be validated against their expected types and required fields.

## Performance analysis

The target is:
```

    90%+ correct function selection.
    90%+ correct argument extraction.
```

This program reached an accuracy of 90.9% for the given function definitions set and 
took approximately 2 minutes and 2 seconds to complete execution.


## Challenges faced

The main challenge of this project was generating regular expressions reliably.
Unlike ordinary strings, regex patterns have their own syntax and special characters that must be handled carefully. The model sometimes generated unnecessarily repeated patterns or included unwanted escape sequences. 

## Testing strategy

The application was tested using individual components and the provided evaluation dataset. Tests covered function selection, argument generation, different parameter types, and edge cases. Generated outputs were compared against expected results to evaluate correctness and reliability.

## Example usage

For example, given the prompt "Add 5 and 10", the application should select fn_add_numbers and generate the arguments {"a": 5, "b": 10}.

## AI Usage

AI was used to assist with the logic of argument constrained decoding only. 
