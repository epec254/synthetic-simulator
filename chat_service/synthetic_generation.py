from databricks.rag_eval.entities import Document
from databricks.rag_eval.context import RealContext
import databricks
from databricks.rag_eval.datasets.managed_evals import _get_managed_evals_client
from typing import Dict, List

# Initialize Databricks RAG eval client
databricks.rag_eval.context._context_singleton = RealContext()
evals_client = _get_managed_evals_client()


def generate_next_question_using_context_from_previous_turn(
    context_from_last_chat_turn: str,
    previous_questions: List[str],
    agent_description: str = None,
) -> List[Dict[str, str]]:
    """
    Question generator using Databricks RAG evaluation.

    Args:
        context_from_last_chat_turn: Context from the last chat turn
        previous_questions: List of previous questions
        num_questions: Number of questions to generate
        agent_description: Optional description of the agent's role

    Returns:
        List of dictionaries containing questions and their context
    """

    question_guidelines = f"""Generate a question based on the provided context.  Questions should be short and in natural language as if a user wrote them.  For example, 
Avoid this list of previously asked questions:
{previous_questions}"""

    doc = Document(content=context_from_last_chat_turn, doc_uri="test.txt")
    questions = evals_client.generate_questions(
        doc=doc,
        num_questions=1,
        agent_description=agent_description,
        question_guidelines=question_guidelines,
    )

    # Convert to expected format
    return [
        {
            "question": q.question,
            "source_doc_uri": q.source_doc_uri,
            "source_context": q.source_context,
        }
        for q in questions
    ]
