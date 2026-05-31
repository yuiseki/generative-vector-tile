from generative_vector_tile.datasets.base import Column, Dataset
from generative_vector_tile.datasets.buildings import buildings
from generative_vector_tile.datasets.countries import economics, population
from generative_vector_tile.datasets.places import places
from generative_vector_tile.datasets.transportation import transportation

REGISTRY: dict[str, Dataset] = {
    places.id: places,
    buildings.id: buildings,
    transportation.id: transportation,
    population.id: population,
    economics.id: economics,
}


def get_dataset(dataset_id: str) -> Dataset:
    if dataset_id not in REGISTRY:
        raise KeyError(f"unknown dataset {dataset_id!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[dataset_id]


def list_datasets() -> list[Dataset]:
    return list(REGISTRY.values())


__all__ = ["Column", "Dataset", "REGISTRY", "get_dataset", "list_datasets"]
