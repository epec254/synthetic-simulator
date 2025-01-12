"""
Example usage of the ChatService with mock server implementations.
"""

import sys
import os
import time
from chat_service import ChatService
from mock_server.server import (
    generate_questions,
    chat_completions,
    ChatCompletionRequest,
    QuestionGenerationRequest,
    Document
)
from pathlib import Path
from typing import Dict, Any, List

def mock_chat_completion(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Mock chat completion using the mock server's implementation directly.
    """
    request = ChatCompletionRequest(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return chat_completions(request)

def mock_question_generator(
    content: str,
    num_questions: int,
    agent_description: str = None,
    question_guidelines: str = None
) -> List[Dict[str, str]]:
    """
    Mock question generator using the mock server's implementation directly.
    """
    request = QuestionGenerationRequest(
        doc=Document(content=content, doc_uri="mock_test.txt"),
        num_questions=num_questions,
        agent_description=agent_description,
        question_guidelines=question_guidelines
    )
    return generate_questions(request)

def main():
    # Create output directory
    output_dir = Path("output")
    output_file = output_dir / "mock_conversation_history.jsonl"

    # Create chat service instance with mock implementations
    chat_service = ChatService(
        chat_agent_callable=mock_chat_completion,
        question_generator_callable=mock_question_generator,
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
