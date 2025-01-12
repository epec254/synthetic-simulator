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

# Load environment variables
load_dotenv()

# Initialize Databricks RAG eval client
databricks.rag_eval.context._context_singleton = RealContext()
evals_client = _get_managed_evals_client()


def example_chat_completion(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Example implementation of a chat completion API.
    Replace this with your actual API implementation.

    Args:
        messages: List of message dictionaries with 'role' and 'content'

    Returns:
        Dict following OpenAI's chat completion response format
    """
    # Get the last user message
    last_message = next(
        (msg for msg in reversed(messages) if msg["role"] == "user"), None
    )
    if not last_message:
        raise ValueError("No user message found")

    # Generate a more detailed response for context extraction
    response = f"This is a real response to: {last_message['content']}. "
    response += "The pants are indeed in the basement, and they are a vibrant shade of green. "
    response += "They are hanging on a wooden rack near the washing machine."

    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": response,
                }
            }
        ]
    }


def example_question_generator(
    content: str,
    num_questions: int,
    agent_description: str = None,
    question_guidelines: str = None,
) -> List[Dict[str, str]]:
    """
    Question generator using Databricks RAG evaluation.

    Args:
        content: Text content to generate questions from
        num_questions: Number of questions to generate
        agent_description: Optional description of the agent's role
        question_guidelines: Optional guidelines for question generation

    Returns:
        List of dictionaries containing questions and their context
    """
    doc = Document(content=content, doc_uri="test.txt")
    questions = evals_client.generate_questions(
        doc=doc,
        num_questions=num_questions,
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


def get_context_from_chat_agent_response(response_data: Dict[str, Any]) -> str:
    """
    Extract context from chat agent response for question generation.

    Args:
        response_data: Response data from chat agent

    Returns:
        str: Extracted context for generating next questions
    """
    # Get the assistant's message content
    content = response_data["choices"][0]["message"]["content"]
    
    # For this example, we'll use the entire response as context
    # In a real implementation, you might want to process or extract specific information
    return content


def main():
    # Create output directory
    output_dir = Path("output")
    output_file = output_dir / "conversation_history.jsonl"

    # Create chat service instance with callable implementations
    chat_service = ChatService(
        chat_agent_callable=example_chat_completion,
        question_generator_callable=example_question_generator,
        get_context_from_chat_agent_response=get_context_from_chat_agent_response,
        max_turns=3,
        model="gpt-3.5-turbo",
        temperature=0.7,
        initial_content="The pants are located in the basement and they are green",
        output_file=str(output_file),
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
