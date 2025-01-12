"""
Example usage of the ChatService with custom API implementations.
"""

import os
from typing import Callable, Dict, Any, List
from chat_service import ChatService
from pathlib import Path
import databricks
from databricks.rag_eval.datasets.managed_evals import _get_managed_evals_client
from databricks.rag_eval.entities import Document
from databricks.rag_eval.context import RealContext
from dotenv import load_dotenv
import mlflow
from mlflow.models.resources import DatabricksServingEndpoint

# Load environment variables
load_dotenv()

# Initialize Databricks RAG eval client
databricks.rag_eval.context._context_singleton = RealContext()
evals_client = _get_managed_evals_client()


# def example_chat_completion(messages: List[Dict[str, str]]) -> Dict[str, Any]:
#     """
#     Example implementation of a chat completion API using MLflow model.
#     Replace this with your actual API implementation.

#     Args:
#         messages: List of message dictionaries with 'role' and 'content'

#     Returns:
#         Dict containing both model outputs and trace information
#     """
#     # Get the last user message
#     last_message = next(
#         (msg for msg in reversed(messages) if msg["role"] == "user"), None
#     )
#     if not last_message:
#         raise ValueError("No user message found")

#     # Load and invoke the model
#     outputs, output_trace = load_and_invoke_model(
#         model_info=DEFAULT_CONFIG["model_uri"],
#         model_input={"messages": messages},
#     )

#     return {
#         "outputs": outputs,
#         "output_trace": output_trace,
#     }


def generate_next_question_from_tool_outputs(
    context_from_last_chat_turn: str,
    previous_questions: List[str],
    agent_description: str = None,
) -> List[Dict[str, str]]:
    """
    Question generator using Databricks RAG evaluation.

    Args:
        context_from_last_chat_turn: Context from the last chat turn
        previous_questions: List of previous questions
        num_questions: Number of questions to generate
        agent_description: Optional description of the agent's role

    Returns:
        List of dictionaries containing questions and their context
    """

    question_guidelines = f"""Generate a question based on the provided context.
Avoid this list of previously asked questions:
{previous_questions}"""

    doc = Document(content=context_from_last_chat_turn, doc_uri="test.txt")
    questions = evals_client.generate_questions(
        doc=doc,
        num_questions=1,
        agent_description=agent_description,
        question_guidelines=question_guidelines,
    )

    # Convert to expected format
    return [
        {
            "question": q.question,
            "source_doc_uri": q.source_doc_uri,
            "source_context": q.source_context,
        }
        for q in questions
    ]


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


def get_all_tool_outputs(response_data: Dict[str, Any]) -> str:
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


from fc_agent import DEFAULT_CONFIG
from model_utils import invoke_model_with_trace

import mlflow
import mlflow.deployments
import mlflow.pyfunc.context as pyfunc_context
import mlflow.tracing.fluent
import mlflow.utils.logging_utils
import uuid

import re

_FAIL_TO_GET_TRACE_WARNING_MSG = re.compile(
    r"Failed to get trace from the tracking store"
)


def log_model(
    agent_code_file: str, agent_config: dict
) -> mlflow.models.model.ModelInfo:
    """Log the model to MLflow and return the model info.

    Args:
        agent_code_file: Path to the agent code file
        agent_config: Configuration dictionary for the agent

    Returns:
        ModelInfo: Information about the logged model
    """
    return mlflow.pyfunc.log_model(
        python_model=agent_code_file,
        artifact_path="agent",
        model_config=agent_config,
        resources=[
            DatabricksServingEndpoint(endpoint_name=agent_config["endpoint_name"])
        ],
        input_example={
            "messages": [{"role": "user", "content": "What is lakehouse monitoring?"}]
        },
        pip_requirements=[
            "databricks-sdk[openai]",
            "mlflow",
            "databricks-agents",
            "backoff",
        ],
    )


def build_chat_completion(
    model_info: mlflow.models.model.ModelInfo,
) -> Callable[[List[Dict[str, str]]], Dict[str, Any]]:
    """
    Build a chat completion function that uses a logged MLflow model.

    Args:
        model_info: ModelInfo object returned from log_model()

    Returns:
        Callable that implements chat completion API using the logged model
    """
    # Load the model once when creating the completion function
    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

    def chat_completion(messages: List[Dict[str, str]]) -> Dict[str, Any]:
        # print(messages)
        # Use invoke_model_uri_with_trace with the cached model's URI
        outputs, output_trace = invoke_model_with_trace(
            model=loaded_model,
            model_input={"messages": messages},
        )

        return {
            "outputs": outputs,
            "output_trace": output_trace,
        }

    return chat_completion


def main():
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "conversation_history.jsonl"

    model_info = log_model(
        agent_code_file=str(Path(__file__).parent / "fc_agent.py"),
        agent_config=DEFAULT_CONFIG,
    )

    chat_completion_callable = build_chat_completion(model_info)

    # # invoke the model
    # model_input = {
    #     "messages": [{"role": "user", "content": "What is lakehouse monitoring?"}]
    # }

    # raw_model_output, output_trace = load_and_invoke_model(model_info, model_input)

    # print(f"Model URI: {model_info.model_uri}")
    # print(f"Trace: {output_trace}")
    # print(f"Output: {raw_model_output}")

    # exit()
    # Create chat service instance with callable implementations
    chat_service = ChatService(
        chat_agent_callable=chat_completion_callable,
        question_generator_callable=generate_next_question_from_tool_outputs,
        get_context_from_chat_agent_response_for_next_turn_callable=get_all_tool_outputs,
        max_turns=5,
        seed_question="what is lakehouse monitoring?",
        output_file=str(output_file),
        agent_description="A chat agent that answers questions about Databricks documentation.",
    )

    # Start the conversation
    try:
        chat_service.start_conversation()
        print(f"\nConversation history saved to: {output_file}")
    except KeyboardInterrupt:
        print("\nStopping chat service...")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
