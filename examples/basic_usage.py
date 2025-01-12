"""
Example usage of the ChatService with custom API implementations.
"""

from typing import Callable, Dict, Any, List
from chat_service import ChatService
from pathlib import Path

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
    last_message = next((msg for msg in reversed(messages) if msg["role"] == "user"), None)
    if not last_message:
        raise ValueError("No user message found")

    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": f"This is a real response to: {last_message['content']}"
            }
        }]
    }

def example_question_generator(
    content: str,
    num_questions: int,
    agent_description: str = None,
    question_guidelines: str = None
) -> List[Dict[str, str]]:
    """
    Example implementation of a question generation API.
    Replace this with your actual API implementation.
    """
    # Simple example that generates numbered questions
    return [
        {
            "question": f"Question {i + 1} about: {content}?",
            "source_doc_uri": "example.txt",
            "source_context": content
        }
        for i in range(num_questions)
    ]

def main():
    # Create output directory
    output_dir = Path("output")
    output_file = output_dir / "conversation_history.jsonl"

    # Create chat service instance with callable implementations
    chat_service = ChatService(
        chat_agent_callable=example_chat_completion,
        question_generator_callable=example_question_generator,
        max_turns=3,
        model="gpt-3.5-turbo",
        temperature=0.7,
        content_for_questions="The pants are located in the basement and they are green",
        output_file=str(output_file)
    )
    
    # Start the conversation
    try:
        chat_service.start_conversation()
        print(f"\nConversation history saved to: {output_file}")
    except KeyboardInterrupt:
        print("\nStopping chat service...")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == '__main__':
    main()
