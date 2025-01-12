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
        question_generator_callable: Callable[[str, str], List[Dict[str, str]]],
        get_context_from_chat_agent_response_for_next_turn_callable: Callable[
            [Dict[str, Any]], str
        ],
        max_turns: int,
        seed_question: str,
        output_file: str = "conversation_history.jsonl",
        agent_description: Optional[str] = None,
    ):
        """
        Initialize the ChatService.

        Args:
            chat_agent_callable: Function that takes messages array and returns chat completion response
            question_generator_callable: Function that generates next question based on context and agent description
            get_context_from_chat_agent_response: Function that extracts context from chat agent response
            max_turns: Maximum number of conversation turns
            seed_question: Initial question to start the conversation
            output_file: Path to save the conversation history
            agent_description: Optional description for question generation
        """
        self.chat_agent_callable = chat_agent_callable
        self.question_generator_callable = question_generator_callable
        self.get_context_from_chat_agent_response = (
            get_context_from_chat_agent_response_for_next_turn_callable
        )
        self.max_turns = max_turns
        self.seed_question = seed_question
        self.output_file = output_file
        self.agent_description = agent_description
        self.conversation_history: List[Dict[str, str]] = []
        self.question_history: List[str] = []
        self.last_chat_response = None

    def save_conversation_turn(self):
        """Save the current conversation turn to the JSONL file."""
        with open(self.output_file, "a") as f:
            # Create a record with just the messages array
            record = {"messages": self.conversation_history.copy()}
            json.dump(record, f)
            f.write("\n")

    def generate_next_question(self) -> str:
        """
        Generate the next question based on the last conversation turn.
        If this is the first turn, return the seed question.

        Returns:
            str: The next question to ask
        """
        if not self.question_history:
            next_question = self.seed_question
            self.question_history.append(next_question)
        else:
            # Get context from the last response
            last_response = self.last_chat_response
            context = self.get_context_from_chat_agent_response(last_response)

            # Generate next question based on the context
            questions = self.question_generator_callable(
                context_from_last_chat_turn=context,
                previous_questions=self.question_history,
                agent_description=self.agent_description,
            )
            next_question = questions[0]["question"]

        self.question_history.append(next_question)
        return next_question

    def call_chat_agent(self, question: str) -> Dict[str, Any]:
        """
        Send a question to the chat agent callable and get the response.

        Args:
            question: The question to ask the chat agent

        Returns:
            Dict[str, Any]: Dictionary containing the chat agent's response text and output trace

        Raises:
            Exception: If the chat completion fails
        """
        # Add the new question to conversation history
        self.conversation_history.append({"role": "user", "content": question})

        # Call the chat completion function with just the messages
        response = self.chat_agent_callable(self.conversation_history)

        # Extract the message content and trace
        assistant_message = response["outputs"]["choices"][0]["message"]["content"]
        output_trace = response["output_trace"]

        # Add the assistant's response to conversation history
        self.conversation_history.append(
            {"role": "assistant", "content": assistant_message}
        )

        # Store the full response for context extraction
        self.last_chat_response = response

        # Save this turn to the JSONL file
        self.save_conversation_turn()

        return {"message": assistant_message, "trace": output_trace}

    def start_conversation(self) -> None:
        """
        Start a conversation loop for the specified number of turns.
        """
        # Create output directory if it doesn't exist
        output_dir = Path(self.output_file).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Add initial system message
        self.conversation_history = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant engaging in a conversation.",
            }
        ]

        # Handle first turn with seed question separately
        try:
            question = self.generate_next_question()  # This will return seed question
            print(f"Initial turn: Asking seed question: {question}")
            response_and_trace = self.call_chat_agent(question)
            print(f'Answer: {response_and_trace["message"]}')
        except Exception as e:
            print(f"Error during initial turn: {str(e)}")
            return

        # Continue with remaining turns
        for turn in range(self.max_turns):
            try:
                question = self.generate_next_question()
                print(f"Turn {turn + 1}: Asking question: {question}")
                response_and_trace = self.call_chat_agent(question)
                print(f'Answer: {response_and_trace["message"]}')
            except Exception as e:
                print(f"Error during turn {turn + 1}: {str(e)}")
                break
