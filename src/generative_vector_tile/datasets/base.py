from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ColumnType = Literal["int", "float", "string", "geometry"]


@dataclass(frozen=True)
class Column:
    name: str
    type: ColumnType
    sql_expr: str
    filterable: bool = False
    # When the column has a closed enum (e.g. Overture buildings `class`),
    # list the legal values here. They're injected into the LLM system
    # prompt so the model emits real values instead of hallucinating
    # plausible-looking strings that miss the data.
    enum_values: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Dataset:
    """Static registration of an Overture-backed source.

    The dataset declares its read_parquet root, output columns (projected into
    the MVT), and which columns are exposed to the `q=` filter pipeline. The
    filter parser will reject any reference to a column that isn't listed in
    `filterable_columns`, so this dataclass is the security boundary for
    column-level access.
    """

    id: str
    description: str
    overture_release: str
    overture_theme: str
    overture_type: str
    columns: tuple[Column, ...]
    mvt_layer_name: str
    filter_aliases: dict[str, str] = field(default_factory=dict)

    @property
    def filterable_columns(self) -> tuple[Column, ...]:
        return tuple(c for c in self.columns if c.filterable)

    @property
    def projected_columns(self) -> tuple[Column, ...]:
        return tuple(c for c in self.columns if c.type != "geometry")

    @property
    def geometry_column(self) -> Column:
        for c in self.columns:
            if c.type == "geometry":
                return c
        raise RuntimeError(f"dataset {self.id} has no geometry column")
