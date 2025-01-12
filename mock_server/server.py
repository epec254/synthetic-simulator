"""
Mock server to simulate chat agent and question API endpoints using OpenAI's Chat Completion API spec.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
import random
import uvicorn
from datetime import datetime

app = FastAPI()

# Sample content for generating questions
SAMPLE_CONTENT = {
    "The pants are located in the basement and they are green": [
        ("Where are the pants located?", "The pants are located in the basement and they are green."),
        ("What color are the pants?", "The pants are located in the basement and they are green.")
    ],
    "Python is a popular programming language known for its readability and extensive library support": [
        ("What characteristics make Python popular?", "Python is a popular programming language known for its readability and extensive library support"),
        ("What kind of support does Python offer?", "Python is a popular programming language known for its readability and extensive library support")
    ]
}

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    user: Optional[str] = None

class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str = "stop"

class Usage(BaseModel):
    prompt_tokens: int = Field(default=100)
    completion_tokens: int = Field(default=50)
    total_tokens: int = Field(default=150)

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random.randint(1000, 9999)}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(datetime.now().timestamp()))
    model: str
    choices: List[Choice]
    usage: Usage

class Document(BaseModel):
    content: str
    doc_uri: str

class SyntheticQuestion(BaseModel):
    question: str
    source_doc_uri: str
    source_context: str

class QuestionGenerationRequest(BaseModel):
    doc: Document
    num_questions: int
    agent_description: Optional[str] = None
    question_guidelines: Optional[str] = None

@app.post("/generate_questions")
async def generate_questions(request: QuestionGenerationRequest) -> List[SyntheticQuestion]:
    """Mock endpoint for generating questions from content."""
    if not request.doc.content:
        raise HTTPException(status_code=400, detail="Content is required")
    
    # Use the sample content if it exists, otherwise generate generic questions
    if request.doc.content in SAMPLE_CONTENT:
        questions = SAMPLE_CONTENT[request.doc.content][:request.num_questions]
    else:
        # Generate generic questions if content not in samples
        questions = [
            (f"Generic Question {i} about the content?", request.doc.content)
            for i in range(request.num_questions)
        ]
    
    return [
        SyntheticQuestion(
            question=q,
            source_doc_uri=request.doc.doc_uri,
            source_context=context
        )
        for q, context in questions
    ]

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """Mock endpoint for chat completions API."""
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages are required")
    
    # Get the last user message
    last_message = next((msg for msg in reversed(request.messages) if msg.role == "user"), None)
    if not last_message:
        raise HTTPException(status_code=400, detail="No user message found")
    
    # Generate a simple response
    response_content = f"This is a mock response to: {last_message.content}"
    
    return ChatCompletionResponse(
        model=request.model,
        choices=[
            Choice(
                index=0,
                message=Message(
                    role="assistant",
                    content=response_content
                )
            )
        ],
        usage=Usage()
    )

def start_server():
    """Start the mock server."""
    uvicorn.run(app, host="127.0.0.1", port=8000)

if __name__ == "__main__":
    start_server()
