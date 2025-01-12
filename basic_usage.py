"""
Example usage of the ChatService with custom API implementations.
"""

from dotenv import load_dotenv


# Load environment variables
load_dotenv()


import logging
from typing import Callable, Dict, Any, List
from chat_service import SyntheticDataSimulatorService
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

# Setup logger
logger = logging.getLogger("chat_service.examples")


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


def generate_based_on_tool_outputs(
    max_turns: int,
    model_info: mlflow.models.model.ModelInfo,
    output_file: str,
    seed_question: str,
    agent_description: str,
    tag: str,
):
    """
    Synthetically generates follow-up questions based on the outputs from all called tools.

    Args:
        max_turns: Maximum number of conversation turns
        model_info: ModelInfo object returned from log_model()
        output_file: Path to the output file for conversation history
        seed_question: Initial question to start the conversation
        agent_description: Description of the chat agent
        tag: Tag for the generation type
    """

    logger.info(
        f"Starting synthetic generation with {max_turns} turns based on tool outputs"
    )

    chat_completion_callable = get_agent_callable(model_info)

    chat_service = SyntheticDataSimulatorService(
        chat_agent_callable=chat_completion_callable,
        question_generator_callable=generate_next_question_using_context_from_previous_turn,
        get_context_from_chat_agent_response_for_next_turn_callable=get_all_tool_outputs_from_agent_trace,
        max_turns=max_turns,
        seed_question=seed_question,
        output_file=output_file,
        agent_description=agent_description,
        tag=tag,
    )

    # Start the conversation
    try:
        chat_service.start_conversation()
    except Exception as e:
        logger.error(f"Error during conversation: {e}")
        raise


def generate_based_on_response(
    max_turns: int,
    model_info: mlflow.models.model.ModelInfo,
    output_file: str,
    seed_question: str,
    agent_description: str,
    tag: str,
):
    """
    Synthetically generates follow-up questions based on the agent's last response.

    Args:
        max_turns: Maximum number of conversation turns
        model_info: ModelInfo object returned from log_model()
        output_file: Path to the output file for conversation history
        seed_question: Initial question to start the conversation
        agent_description: Description of the chat agent
        tag: Tag for the generation type
    """

    logger.info(
        f"Starting synthetic generation with {max_turns} turns based on response content"
    )

    chat_completion_callable = get_agent_callable(model_info)

    chat_service = SyntheticDataSimulatorService(
        chat_agent_callable=chat_completion_callable,
        question_generator_callable=generate_next_question_using_context_from_previous_turn,
        get_context_from_chat_agent_response_for_next_turn_callable=get_agent_response_from_trace,
        max_turns=max_turns,
        seed_question=seed_question,
        output_file=output_file,
        agent_description=agent_description,
        tag=tag,
    )

    # Start the conversation
    try:
        chat_service.start_conversation()
    except Exception as e:
        logger.error(f"Error during conversation: {e}")
        raise


def main():
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_file = str(output_dir / "synthetic_evaluation_set.jsonl")

    # Log the model
    # We use a function-calling agent defined in fc_agent.py that queries databricks documentation through a keyword-based retriever tool.
    # You can replace this with your own agent implementation.

    # Note: this configuration is specific to the code in the `fc_agent.py` file.  Your own agent will likely have a different configuration.
    agent_config = {
        "endpoint_name": "ep-gpt4o-new",  # replace with a Model Serving endpoint that supports Chat Completions.  Can be an external model e.g., OpenAI
        "temperature": 0.01,
        "max_tokens": 1000,
        "system_prompt": """You are a helpful assistant that answers questions about Databricks. Questions unrelated to Databricks are irrelevant.

    You answer questions using a set of tools. If needed, you ask the user follow-up questions to clarify their request.
    """,
        "max_context_chars": 4096 * 4,
    }
    model_info = log_model(
        agent_code_file=str(Path(__file__).parent / "fc_agent.py"),
        agent_config=agent_config,
    )

    # Common parameters
    params = {
        "max_turns": 2,
        "model_info": model_info,
        "output_file": output_file,
        "seed_question": "what is lakehouse monitoring?",
        "agent_description": "A chat agent that answers questions about Databricks documentation.",
    }

    # Run both types of generation with different tags
    generate_based_on_tool_outputs(**params, tag="tool_outputs")
    generate_based_on_response(**params, tag="agent_response")


if __name__ == "__main__":
    main()
