"""
Example usage of the ChatService with custom API implementations.
"""

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


from typing import Callable, Dict, Any, List
from chat_service import ChatService
from pathlib import Path
import mlflow
from mlflow.models.resources import DatabricksServingEndpoint
from fc_agent import DEFAULT_CONFIG
from model_utils import invoke_model_with_trace
from chat_service.context_generators import (
    get_all_tool_outputs_from_agent_trace,
    get_agent_response_from_trace,
)
from chat_service.synthetic_generation import (
    generate_next_question_using_context_from_previous_turn,
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


def get_agent_callable(
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

    def call_mlflow_logged_agent(messages: List[Dict[str, str]]) -> Dict[str, Any]:
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

    return call_mlflow_logged_agent


def generate_based_on_tool_outputs():
    """
    Synthetically generates follow-up questions based on the outputs from all called tools.
    """
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "conversation_history.jsonl"

    model_info = log_model(
        agent_code_file=str(Path(__file__).parent / "fc_agent.py"),
        agent_config=DEFAULT_CONFIG,
    )

    chat_completion_callable = get_agent_callable(model_info)

    chat_service = ChatService(
        chat_agent_callable=chat_completion_callable,
        question_generator_callable=generate_next_question_using_context_from_previous_turn,
        get_context_from_chat_agent_response_for_next_turn_callable=get_all_tool_outputs_from_agent_trace,
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


def generate_based_on_response():
    """
    Synthetically generates follow-up questions based on the agent's last response.
    """
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "conversation_history.jsonl"

    model_info = log_model(
        agent_code_file=str(Path(__file__).parent / "fc_agent.py"),
        agent_config=DEFAULT_CONFIG,
    )

    chat_completion_callable = get_agent_callable(model_info)

    chat_service = ChatService(
        chat_agent_callable=chat_completion_callable,
        question_generator_callable=generate_next_question_using_context_from_previous_turn,
        get_context_from_chat_agent_response_for_next_turn_callable=get_agent_response_from_trace,
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


def main():
    # generate_based_on_tool_outputs()
    generate_based_on_response()


if __name__ == "__main__":
    main()
