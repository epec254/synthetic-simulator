"""
Example usage of the ChatService with mock server.
"""

import sys
import os
import time
from chat_service import ChatService
from mock_server.server import start_server
import threading
import requests
from pathlib import Path

def wait_for_server(url: str, max_retries: int = 5, retry_delay: float = 1.0):
    """Wait for the server to start up."""
    for _ in range(max_retries):
        try:
            requests.get(url)
            return True
        except requests.ConnectionError:
            time.sleep(retry_delay)
    return False

def main():
    # Start mock server in a separate thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    # Wait for the server to start
    if not wait_for_server("http://127.0.0.1:8000/generate_questions"):
        print("Failed to start mock server")
        sys.exit(1)

    # Create output directory
    output_dir = Path("output")
    output_file = output_dir / "conversation_history.jsonl"

    # Create chat service instance with mock server URLs
    chat_service = ChatService(
        chat_agent_url='http://127.0.0.1:8000/v1/chat/completions',
        question_api_url='http://127.0.0.1:8000',
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
