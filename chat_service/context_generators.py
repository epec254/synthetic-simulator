from typing import Dict, Any
import re


def get_tool_call_outputs_as_sent_to_llm(trace):
    """
    For each LLM call w/ requested tools, get all available tools + tool that were requested
    """
    pattern = re.compile(r"^chat_completions_api_\d+$")
    llm_calls = trace.search_spans(name=pattern)
    tool_outputs = []
    for llm_call in llm_calls:
        # print(llm_call.outputs)
        input_messages = llm_call.inputs["messages"]

        for input_message in input_messages:
            if input_message["role"] == "tool":
                tool_outputs.append(input_message["content"])

    return tool_outputs


def get_all_tool_outputs_from_agent_trace(response_data: Dict[str, Any]) -> str:
    """
    Extract context from chat agent response for question generation.

    Args:
        response_data: Response data from chat agent containing both outputs and trace

    Returns:
        str: Extracted context for generating next questions
    """
    # Get the assistant's message content from outputs
    outputs = response_data["outputs"]
    output_trace = response_data["output_trace"]

    tool_outputs = get_tool_call_outputs_as_sent_to_llm(output_trace)
    # print(f"Tool outputs: {tool_outputs}")

    # Extract message content
    content = outputs["choices"][0]["message"]["content"]

    # For this example, we'll use the entire response as context
    # In a real implementation, you might want to:
    # 1. Process the content based on trace information
    # 2. Add trace metadata to the context
    # 3. Filter or enhance the context based on model performance metrics
    return "\n".join(tool_outputs)
