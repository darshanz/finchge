from dataclasses import dataclass, field
from typing import Any


@dataclass
class Fitness:
    value: float
    case_data: dict[str, list[float]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)



@dataclass
class EvaluationRecord:
    fitness: list[float]
    case_data: dict[str, list[float]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)



def merge_fitness_results(results: list[Fitness]) -> EvaluationRecord:
    fitness_values: list[float] = []
    case_data: dict[str, list[float]] = {}
    meta: dict[str, Any] = {}

    for result in results:
        fitness_values.append(result.value)
        for key, values in result.case_data.items():
            if key in case_data:
                raise ValueError(f"Duplicate case_data key '{key}'")
            case_data[key] = values
        meta.update(result.meta)

    return EvaluationRecord(
        fitness=fitness_values,
        case_data=case_data,
        meta=meta,
    )