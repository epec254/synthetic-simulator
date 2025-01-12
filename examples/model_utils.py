"""Helper functions for model invocation and trace handling."""

import re
import uuid
from typing import Any, Tuple, Union

import mlflow
import mlflow.pyfunc
import mlflow.pyfunc.context as pyfunc_context
import mlflow.tracing.fluent
import mlflow.utils.logging_utils
from mlflow.models.model import ModelInfo

_FAIL_TO_GET_TRACE_WARNING_MSG = re.compile(
    r"Failed to get trace from the tracking store"
)


def invoke_model_with_trace(
    model: mlflow.pyfunc.PyFuncModel,
    model_input: Any,
) -> Tuple[Any, Any]:
    """
    Invoke a model and capture its trace information.

    Args:
        model: The MLflow model to invoke
        model_input: Input to pass to the model

    Returns:
        Tuple containing (model_output, trace)
    """
    context_id = str(uuid.uuid4())
    with pyfunc_context.set_prediction_context(
        pyfunc_context.Context(context_id, is_evaluate=True)
    ), mlflow.utils.logging_utils.suppress_logs(
        mlflow.tracing.fluent.__name__, _FAIL_TO_GET_TRACE_WARNING_MSG
    ):
        raw_model_output = model.predict(model_input)
        output_trace = mlflow.get_trace(context_id)

    return raw_model_output, output_trace


def invoke_model_uri_with_trace(
    model_uri: str,
    model_input: Any,
) -> Tuple[Any, Any]:
    """
    Load a model from a URI and invoke it with trace capture.

    Args:
        model_uri: URI of the MLflow model to load and invoke
        model_input: Input to pass to the model

    Returns:
        Tuple containing (model_output, trace)
    """
    model = mlflow.pyfunc.load_model(model_uri)
    return invoke_model_with_trace(model, model_input)


def load_and_invoke_model(
    model_info: Union[ModelInfo, str],
    model_input: Any,
) -> Tuple[Any, Any]:
    """
    Load a model from ModelInfo or URI and invoke it with trace capture.

    Args:
        model_info: Either a ModelInfo object from log_model() or a model URI string
        model_input: Input to pass to the model

    Returns:
        Tuple containing (model_output, trace)
    """
    model_uri = (
        model_info.model_uri if isinstance(model_info, ModelInfo) else model_info
    )
    return invoke_model_uri_with_trace(model_uri, model_input)
