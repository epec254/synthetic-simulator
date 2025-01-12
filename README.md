# Chat Service

A multi-turn chat service that interacts with Chat Agent API and generates questions dynamically.

## Features

- Configurable number of conversation turns
- Integration with external Chat Agent API
- Dynamic question generation through Question API
- Error handling and logging
- Type hints for better code maintainability

## Installation

This project uses Poetry for dependency management. To install:

```bash
poetry install
```

## Usage

Basic usage example:

```python
from chat_service import ChatService

chat_service = ChatService(
    chat_agent_url='https://chat-agent-api.example.com',
    question_api_url='https://question-api.example.com',
    max_turns=5
)

chat_service.start_conversation()
```

See the `examples` directory for more usage examples.

## Requirements

- Python ^3.11
- Poetry for dependency management
- `requests` library for API interactions
