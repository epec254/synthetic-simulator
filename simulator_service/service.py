"""
Main service module for handling chat interactions.
"""

from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import json
from .logging_config import logger


class EmptyContextError(Exception):
    """Raised when no context could be extracted from the chat agent's response."""

    pass


class SyntheticDataSimulatorService:
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
        use_last_context_if_cannot_generate_context: bool = False,
        tag: Optional[str] = None,
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
            use_last_context_if_cannot_generate_context: If True, use the last valid context when no new context can be extracted
            tag: Optional tag to add to each saved conversation turn
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
        self.use_last_context_on_empty = use_last_context_if_cannot_generate_context
        self.tag = tag
        self.conversation_history: List[Dict[str, str]] = []
        self.question_history: List[str] = []
        self.last_chat_response = None
        self.last_valid_context: Optional[str] = None

    def save_conversation_turn(self, first_turn: bool = False):
        """
        Save the current conversation turn to the JSONL file.

        Args:
            tag: Optional tag to add to the output record. If not provided, uses the service's default tag if set.
        """
        with open(self.output_file, "a") as f:
            # Create a record with messages array (excluding last assistant message) and separate assistant response
            messages = (
                self.conversation_history[:-1].copy()
                if self.conversation_history
                else []
            )
            record = {
                "request": messages,
                "response": (
                    self.conversation_history[-1] if self.conversation_history else None
                ),
                "metadata": {
                    "context_used_to_generate_question": self.last_valid_context,
                },
            }

            # Add special tag for first turn based on seed question
            if first_turn:
                record["metadata"]["first_turn"] = True
                record["metadata"]["seed_question"] = self.seed_question
            else:
                record["metadata"]["first_turn"] = False

            # add tag if provided
            if self.tag:
                record["metadata"]["tag"] = self.tag
            json.dump(record, f)
            f.write("\n")

    def generate_next_question(self) -> str:
        """
        Generate the next question based on the last conversation turn.
        If this is the first turn, return the seed question.

        Returns:
            str: The next question to ask

        Raises:
            EmptyContextError: If no context could be extracted from the last response and use_last_context_if_cannot_generate_context is False,
                             or if no previous context exists when use_last_context_if_cannot_generate_context is True
        """
        if not self.question_history:
            next_question = self.seed_question
            self.question_history.append(next_question)
            return next_question

        # Get context from the last response
        last_response = self.last_chat_response
        context = self.get_context_from_chat_agent_response(last_response)

        # Check if context is empty
        if not context:
            if self.use_last_context_on_empty and self.last_valid_context:
                context = self.last_valid_context
            else:
                error_msg = (
                    "The last response had no context to generate another question from. "
                    if self.last_valid_context is not None
                    else "No context could be generated from the first response, and no previous context is available. "
                )
                error_msg += "Make sure your `get_context_from_chat_agent_response_for_next_turn_callable` is able to always return a context. "
                error_msg += "This might be because you are relying on outputs in the trace that weren't present for some turns e.g., "
                error_msg += (
                    "you used tool outputs and the model didn't need to call any tools"
                )
                raise EmptyContextError(error_msg)

        # Store the last valid context
        if context:
            self.last_valid_context = context

        # Generate next question based on the context
        questions = self.question_generator_callable(
            context_from_last_chat_turn=context,
            previous_questions=self.question_history,
            agent_description=self.agent_description,
        )
        next_question = questions[0]["question"]
        self.question_history.append(next_question)

        return next_question

    def call_chat_agent(
        self, question: str, first_turn: bool = False
    ) -> Dict[str, Any]:
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
        self.save_conversation_turn(first_turn)

        return {"message": assistant_message, "trace": output_trace}

    def start_conversation(self) -> None:
        """
        Start a conversation loop for the specified number of turns.
        Stops if an error occurs or if no context can be generated.
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
            logger.info(f"Initial turn: Asking seed question: {question}")
            response_and_trace = self.call_chat_agent(question, first_turn=True)
            logger.info(f'Answer: {response_and_trace["message"]}')
        except Exception as e:
            logger.error(f"Error during initial turn: {str(e)}")
            return

        # Continue with remaining turns
        for turn in range(self.max_turns):
            try:
                question = self.generate_next_question()
                logger.info(f"Turn {turn + 1}: Asking question: {question}")
                response_and_trace = self.call_chat_agent(question)
                logger.info(f'Answer: {response_and_trace["message"]}')
            except EmptyContextError as e:
                logger.warning(f"Conversation stopped: {str(e)}")
                break
            except Exception as e:
                logger.error(f"Error during turn {turn + 1}: {str(e)}")
                break
