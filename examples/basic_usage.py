"""
Example usage of the ChatService.
"""

from chat_service import ChatService

def main():
    # Example configuration
    chat_service = ChatService(
        chat_agent_url='https://chat-agent-api.example.com',
        question_api_url='https://question-api.example.com',
        max_turns=5
    )
    
    # Start the conversation
    chat_service.start_conversation()

if __name__ == '__main__':
    main()
