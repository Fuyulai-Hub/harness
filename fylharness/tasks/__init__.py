"""Built-in evaluation tasks.

Importing this package registers the three default task types so they are available by
name in the config file without the user needing to import them explicitly:

* ``multiple_choice`` -- few-shot MCQ (MMLU-style) with letter answers.
* ``text_generation`` -- open-ended generation scored with BLEU/ROUGE.
* ``open_qa``         -- short-answer QA scored with exact match / F1.

Custom tasks are added with :func:`~fylharness.register_task`.
"""

from .base import (
    Task,
    register_task,
    get_task,
    list_tasks,
    task_registry,
)
from .multiple_choice import MultipleChoiceTask
from .text_generation import TextGenerationTask
from .open_qa import OpenQATask

__all__ = [
    "Task",
    "register_task",
    "get_task",
    "list_tasks",
    "task_registry",
    "MultipleChoiceTask",
    "TextGenerationTask",
    "OpenQATask",
]
