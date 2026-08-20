from generative_vector_tile.datasets.base import Column, Dataset
from generative_vector_tile.stac_index import LATEST_RELEASE

OVERTURE_RELEASE = LATEST_RELEASE

places = Dataset(
    id="places",
    description="Overture Places - POIs with category, name, confidence.",
    overture_release=OVERTURE_RELEASE,
    overture_theme="places",
    overture_type="place",
    mvt_layer_name="places",
    columns=(
        Column(name="id", type="string", sql_expr="id"),
        Column(name="geom", type="geometry", sql_expr="geometry"),
        Column(
            name="name",
            type="string",
            sql_expr="names.primary",
            filterable=True,
        ),
        Column(
            name="category",
            type="string",
            sql_expr="categories.primary",
            filterable=True,
        ),
        Column(
            name="confidence",
            type="float",
            sql_expr="confidence",
            filterable=True,
        ),
    ),
    filter_aliases={
        "カテゴリ": "category",
        "種別": "category",
        "名前": "name",
        "名称": "name",
        "信頼度": "confidence",
    },
)
