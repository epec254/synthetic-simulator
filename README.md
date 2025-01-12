# Chat Service

A Python utility for generating dynamic multi-turn conversations using [Agent Evaluation's Synthetic Generation API](https://docs.databricks.com/en/generative-ai/agent-evaluation/synthesize-evaluation-set.html). This utility simulates user interactions with your Agent's model using Agent Evaluation's Synthetic Generation API to dynamically generate follow-up questions.

## Overview

This service is designed for:
- Generating synthetic evaluation data
- Testing agents in a multi-turn setting
- Simulating user interactions with AI agents

The service maintains conversation state and handles the interaction between:
- **Agent's Logged MLflow Model**: Provides AI-generated responses (hosted on Databricks)
- **Agent Evaluation Synthetic Data Ganeration API**: Generates contextually relevant follow-up questions

## Prerequisites

Before you begin, ensure you have:
- Python 3.11 or higher installed
- A Databricks workspace account
- Poetry (Python package manager)
- Basic understanding of MLflow

### Installing Poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Verify the installation:
```bash
poetry --version
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd chat-service
```

2. Install dependencies using Poetry:
```bash
poetry install
```

## Configuration

### 1. Databricks Setup

1. Log into your Databricks workspace
2. Generate a Personal Access Token:
   - Go to User Settings → Developer
   - Click "Manage Access Tokens" → "Generate New Token"
   - Save the token securely

### 2. Environment Variables

Copy the sample environment file:
```bash
cp .env.sample .env
```

Update `.env` with your values:
```bash
# Databricks Configuration
DATABRICKS_TOKEN=<your-databricks-token>    # The PAT token generated above
DATABRICKS_HOST=<workspace-url>             # e.g., https://your-org.cloud.databricks.com

# MLflow Configuration
MLFLOW_TRACKING_URI=databricks              # Keep as 'databricks' for Databricks integration
MLFLOW_EXPERIMENT_NAME=/Users/<email>/chat-service-experiment  # Your MLflow experiment path
```

> The MLflow experiment path is used to store / access the the logged Agent model and store the MLflow Traces generated during the simulation.

## Usage

The easiest way to get started is to use the example in `examples/basic_usage.py`. This example shows how to:
1. Log your agent as an MLflow model
2. Create the required callables using provided utility functions
3. Set up and run the synthetic data generation

Here's a simplified version of the example:

```python
from chat_service import SyntheticDataSimulatorService
import mlflow
from chat_service.context_generators import get_all_tool_outputs_from_agent_trace
from chat_service.synthetic_generation import generate_next_question_using_context_from_previous_turn
from model_utils import invoke_model_with_trace

# 1. Log your agent as an MLflow model
model_info = mlflow.pyfunc.log_model(
    python_model="path/to/agent.py",
    artifact_path="agent",
    # ... other configuration
)

# 2. Create a chat agent callable using the logged model
def chat_agent_callable(messages):
    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    outputs, output_trace = invoke_model_with_trace(
        model=loaded_model,
        model_input={"messages": messages},
    )
    return {
        "outputs": outputs,
        "output_trace": output_trace,
    }

# 3. Initialize and run the chat service
chat_service = SyntheticDataSimulatorService(
    chat_agent_callable=chat_agent_callable,
    question_generator_callable=generate_next_question_using_context_from_previous_turn,
    get_context_from_chat_agent_response_for_next_turn_callable=get_all_tool_outputs_from_agent_trace,
    max_turns=5,
    seed_question="What is lakehouse monitoring?",
    output_file="conversation_output.jsonl", #where to save the resulting synthetic data
    agent_description="An AI assistant that helps with Databricks lakehouse monitoring" # helps tune the synthetic data generation to be aware of your agent's domain
)

chat_service.start_conversation()
```

For a complete working example with error handling and additional features, see `examples/basic_usage.py`.

## Examples

The `examples` directory contains:
* `basic_usage.py`: Complete example with Databricks integration
* `fc_agent.py`: Sample function-calling agent implementation
* `model_utils.py`: Utilities for MLflow model invocation and trace capture

## Required Callable Signatures

The service requires three callable functions with specific signatures:

#### 1. chat_agent_callable

```python
def chat_agent_callable(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Function that calls the agent's MLflow model to get a response.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
                 Example: [{"role": "user", "content": "What is lakehouse monitoring?"}]
    
    Returns:
        Dict containing at least:
            - outputs: The model's outputs
            - output_trace: The MLflow trace of the model's execution
    """
```

#### 2. question_generator_callable

> **Note:** You should not need to modify this callable - it's built to work with Databricks' Agent Evaluation Synthetic Generation API. However, you can replace it with your own function if you want to generate the next turn of conversation using a custom LLM call or different approach.

```python
def question_generator_callable(context: str, agent_description: str) -> List[Dict[str, str]]:
    """
    Function that generates the next question using Agent Evaluation API.
    
    Args:
        context: String containing context from the previous turn
        agent_description: Description of the agent's capabilities
    
    Returns:
        List of message dictionaries with 'role' and 'content' keys
        Example: [{"role": "user", "content": "Can you explain more about data quality monitoring?"}]
    """
```

#### 3. get_context_from_chat_agent_response_for_next_turn_callable

This callable extracts data from the agent's last response, which is then used as input to the Synthetic Generation API to generate the next conversation turn. 

We provide two sample implementations in `chat_service.context_generators`:
- `get_agent_response_from_trace`: Uses just the agent's final response text
- `get_all_tool_outputs_from_agent_trace`: Uses the outputs from all tools the agent called, providing richer context

You can implement your own version to customize how context is extracted and influence the synthetic conversation generation.

```python
def get_context_from_chat_agent_response_for_next_turn_callable(
    agent_response: Dict[str, Any]
) -> str:
    """
    Function that extracts context from the agent's response to use as input to the Synthetic API to generate the next user question to ask the agent..
    
    Args:
        agent_response: Dictionary containing the agent's response with outputs and trace
    
    Returns:
        String containing the extracted context for generating the next question
    
    Raises:
        EmptyContextError: If no context could be extracted from the response
    """
```

The service provides default implementations for these callables in:
- `chat_service.context_generators`: Contains `get_all_tool_outputs_from_agent_trace` and `get_agent_response_from_trace`
- `chat_service.synthetic_generation`: Contains `generate_next_question_using_context_from_previous_turn`

See `examples/basic_usage.py` for complete implementation examples.

## Model Utilities

> TODO: Add a version of `invoke_model_with_trace` that works with a Databricks Model Serving endpoint

The service provides utility functions in `examples/model_utils.py` to help with model invocation and MLflow trace capture:

### invoke_model_with_trace

```python
def invoke_model_with_trace(
    model: mlflow.pyfunc.PyFuncModel,
    model_input: Any,
) -> Tuple[Any, Any]:
    """
    Invoke a model and capture its MLflow trace information.
    
    Args:
        model: The MLflow model to invoke (loaded via mlflow.pyfunc.load_model)
        model_input: Input to pass to the model (e.g., {"messages": [...]})
    
    Returns:
        Tuple of (model_output, mlflow_trace)
    """
```

### invoke_model_uri_with_trace

```python
def invoke_model_uri_with_trace(
    model_uri: str,
    model_input: Any,
) -> Tuple[Any, Any]:
    """
    Load a model from a URI and invoke it with trace capture.
    
    Args:
        model_uri: URI of the MLflow model (e.g., "models:/my-model/production")
        model_input: Input to pass to the model
    
    Returns:
        Tuple of (model_output, mlflow_trace)
    """
```

### load_and_invoke_model

```python
def load_and_invoke_model(
    model_info: Union[ModelInfo, str],
    model_input: Any,
) -> Tuple[Any, Any]:
    """
    Load a model from ModelInfo or URI and invoke it with trace capture.
    
    Args:
        model_info: Either a ModelInfo object from log_model() or a model URI string
        model_input: Input to pass to the model
    
    Returns:
        Tuple of (model_output, mlflow_trace)
    """
```

These utilities handle:
- Loading MLflow models
- Setting up the prediction context
- Capturing MLflow traces during model invocation
- Proper cleanup of resources

Example usage in your chat agent callable:

```python
from model_utils import invoke_model_with_trace

def chat_agent_callable(messages):
    # Load your model
    model = mlflow.pyfunc.load_model(model_uri)
    
    # Invoke with trace capture
    outputs, output_trace = invoke_model_with_trace(
        model=model,
        model_input={"messages": messages}
    )
    
    return {
        "outputs": outputs,
        "output_trace": output_trace  # Required for context extraction
    }
```

## Troubleshooting

Common issues and solutions:

1. **Poetry Installation Fails**
   - Ensure Python 3.11+ is installed: `python --version`
   - Try installing Poetry via pip: `pip install poetry`

2. **Databricks Connection Issues**
   - Verify your token is valid and not expired
   - Check if DATABRICKS_HOST includes the full URL with https://
   - Ensure your IP is allowlisted in Databricks workspace settings

3. **MLflow Errors**
   - Confirm MLFLOW_TRACKING_URI is set to "databricks"
   - Verify you have permission to create experiments in the specified path

## API Documentation

### SyntheticDataSimulatorService

Main class for generating synthetic conversations:

```python
SyntheticDataSimulatorService(
    chat_agent_callable: Callable,          # Function that calls the agent's MLflow model
    question_generator_callable: Callable,   # Function that generates questions using Agent Evaluation API
    get_context_from_chat_agent_response_for_next_turn_callable: Callable,  # Function that extracts context from agent response
    max_turns: int,                         # Maximum conversation turns
    seed_question: str,                     # Initial question to start with
    output_file: str = "conversation_history.jsonl",  # Where to save the conversation
    agent_description: Optional[str] = None,          # Description of the agent for question generation
    use_last_context_if_cannot_generate_context: bool = False,  # Fallback behavior for context extraction
    tag: Optional[str] = None               # Optional tag for conversation turns
)
```

For detailed API documentation, see the docstrings in the source code.

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting pull requests.

## License

[Add your license information here]
