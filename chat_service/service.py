"""
Main service module for handling chat interactions.
"""

import requests
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

class ChatService:
    def __init__(
        self,
        chat_agent_url: str,
        question_api_url: str,
        max_turns: int,
        model: str = "gpt-3.5-turbo",
        temperature: float = 1.0,
        content_for_questions: str = "Python is a popular programming language known for its readability and extensive library support",
        output_file: str = "conversation_history.jsonl"
    ):
        """
        Initialize the ChatService.

        Args:
            chat_agent_url: URL endpoint for the chat agent API
            question_api_url: URL endpoint for the question generation API
            max_turns: Maximum number of conversation turns
            model: The model to use for chat completion
            temperature: Sampling temperature between 0 and 2
            content_for_questions: Content to generate questions from
            output_file: Path to save the conversation history
        """
        self.chat_agent_url = chat_agent_url
        self.question_api_url = question_api_url
        self.max_turns = max_turns
        self.model = model
        self.temperature = temperature
        self.content_for_questions = content_for_questions
        self.output_file = output_file
        self.conversation_history: List[Dict[str, str]] = []
        self.current_question_index = 0
        self.questions = []

    def save_conversation_turn(self):
        """Save the current conversation turn to the JSONL file."""
        with open(self.output_file, 'a') as f:
            # Create a record with just the messages array
            record = {
                "messages": self.conversation_history.copy()
            }
            json.dump(record, f)
            f.write('\n')

    def get_next_question(self) -> str:
        """
        Get the next question from the question API.

        Returns:
            str: The generated question
        
        Raises:
            requests.RequestException: If the API call fails
        """
        # If we've used all questions or haven't fetched any yet, get new ones
        if not self.questions or self.current_question_index >= len(self.questions):
            payload = {
                "doc": {
                    "content": self.content_for_questions,
                    "doc_uri": "chat_service.txt"
                },
                "num_questions": self.max_turns
            }
            
            response = requests.post(f"{self.question_api_url}/generate_questions", json=payload)
            response.raise_for_status()
            
            self.questions = response.json()
            self.current_question_index = 0
        
        # Get the next question and increment the index
        question = self.questions[self.current_question_index]["question"]
        self.current_question_index += 1
        return question

    def call_chat_agent(self, question: str) -> str:
        """
        Send a question to the chat agent and get the response.

        Args:
            question: The question to ask the chat agent

        Returns:
            str: The chat agent's response text

        Raises:
            requests.RequestException: If the API call fails
        """
        # Add the new question to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": question
        })

        payload = {
            "model": self.model,
            "messages": self.conversation_history,
            "temperature": self.temperature
        }

        response = requests.post(self.chat_agent_url, json=payload)
        response.raise_for_status()
        
        response_data = response.json()
        assistant_message = response_data["choices"][0]["message"]["content"]
        
        # Add the assistant's response to conversation history
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })

        # Save this turn to the JSONL file
        self.save_conversation_turn()

        return assistant_message

    def start_conversation(self) -> None:
        """
        Start a conversation loop for the specified number of turns.
        """
        # Create output directory if it doesn't exist
        output_dir = Path(self.output_file).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Add initial system message
        self.conversation_history = [{
            "role": "system",
            "content": "You are a helpful AI assistant engaging in a conversation."
        }]

        for turn in range(self.max_turns):
            try:
                question = self.get_next_question()
                print(f'Turn {turn + 1}: Asking question: {question}')
                answer = self.call_chat_agent(question)
                print(f'Answer: {answer}')
            except requests.RequestException as e:
                print(f'Error during turn {turn + 1}: {str(e)}')
