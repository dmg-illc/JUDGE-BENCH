from typing import Dict, Literal, Optional, Union

from pydantic import BaseModel


class GradedAnnotationScores(BaseModel):
    """Scores for graded annotations."""

    mean_human: float
    individual_human_scores: list[int]
    majority_human: Optional[int]


class ContinuousAnnotationScores(BaseModel):
    """Scores for continuous annotations."""

    mean_human: float
    individual_human_scores: list[float]


class CategoricalAnnotationScores(BaseModel):
    """Scores for categorical annotations."""

    majority_human: str
    individual_human_scores: list[str]


"""Represents scores for different types of annotations."""
AnnotationScores = Union[
    GradedAnnotationScores, ContinuousAnnotationScores, CategoricalAnnotationScores
]


class Instance(BaseModel):
    """Represents an instance in an evaluation dataset.

    An instance is a single data point in the evaluation dataset. In case the evaluator has
    to express a judgement regarding a single instance, the instance field should contain a single
    string.
    Otherwise, the instance field should contain a dict, the value associated with each key
    represents a different piece of textual information provided to the evaluator."""

    id: Union[int, str]
    instance: Union[str, dict]
    annotations: Dict[str, AnnotationScores]


class Annotation(BaseModel):
    """Represents the type of annotation in an evaluation dataset."""

    metric: str
    prompt: str
    category: Literal["graded", "categorical", "continuous"]


class GradedAnnotation(Annotation):
    """Represents a graded annotation having a lower and upper bound."""

    worst: int
    best: int


class CategoricalAnnotation(Annotation):
    """Represents a categorical with a list of labels."""

    labels_list: list[str]


class ContinuousAnnotation(Annotation):
    """Represents a continuous annotation having a lower and upper bound."""

    worst: float
    best: float


AnnotationType = Union[GradedAnnotation, CategoricalAnnotation, ContinuousAnnotation]


class Dataset(BaseModel):
    """Represents an evaluation dataset."""

    dataset: str
    dataset_url: str
    annotations: list[AnnotationType]
    instances: list[Instance]
    expert_annotator: Literal["true", "unknown", "false"]
    original_prompt: bool
