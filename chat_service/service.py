"""
Main service module for handling chat interactions.
"""

from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import json

class ChatService:
    def __init__(
        self,
        chat_agent_callable: Callable[[List[Dict[str, str]]], Dict[str, Any]],
        question_generator_callable: Callable[[str, int, Optional[str], Optional[str]], List[Dict[str, str]]],
        max_turns: int,
        model: str = "gpt-3.5-turbo",
        temperature: float = 1.0,
        content_for_questions: str = "Python is a popular programming language known for its readability and extensive library support",
        output_file: str = "conversation_history.jsonl",
        agent_description: Optional[str] = None,
        question_guidelines: Optional[str] = None
    ):
        """
        Initialize the ChatService.

        Args:
            chat_agent_callable: Function that takes messages array and returns chat completion response
            question_generator_callable: Function that implements question generation API
            max_turns: Maximum number of conversation turns
            model: The model to use for chat completion
            temperature: Sampling temperature between 0 and 2
            content_for_questions: Content to generate questions from
            output_file: Path to save the conversation history
            agent_description: Optional description for question generation
            question_guidelines: Optional guidelines for question generation
        """
        self.chat_agent_callable = chat_agent_callable
        self.question_generator_callable = question_generator_callable
        self.max_turns = max_turns
        self.model = model
        self.temperature = temperature
        self.content_for_questions = content_for_questions
        self.output_file = output_file
        self.agent_description = agent_description
        self.question_guidelines = question_guidelines
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
        Get the next question using the question generator callable.

        Returns:
            str: The generated question
        
        Raises:
            Exception: If the question generation fails
        """
        # If we've used all questions or haven't fetched any yet, get new ones
        if not self.questions or self.current_question_index >= len(self.questions):
            self.questions = self.question_generator_callable(
                content=self.content_for_questions,
                num_questions=self.max_turns,
                agent_description=self.agent_description,
                question_guidelines=self.question_guidelines
            )
            self.current_question_index = 0
        
        # Get the next question and increment the index
        question = self.questions[self.current_question_index]["question"]
        self.current_question_index += 1
        return question

    def call_chat_agent(self, question: str) -> str:
        """
        Send a question to the chat agent callable and get the response.

        Args:
            question: The question to ask the chat agent

        Returns:
            str: The chat agent's response text

        Raises:
            Exception: If the chat completion fails
        """
        # Add the new question to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": question
        })

        # Call the chat completion function with just the messages
        response_data = self.chat_agent_callable(self.conversation_history)
        
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
            except Exception as e:
                print(f'Error during turn {turn + 1}: {str(e)}')
