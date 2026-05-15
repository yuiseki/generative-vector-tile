from generative_vector_tile.datasets.base import Column, Dataset

OVERTURE_RELEASE = "2026-04-15.0"

buildings = Dataset(
    id="buildings",
    description="Overture Buildings — building footprints with height and class.",
    overture_release=OVERTURE_RELEASE,
    overture_theme="buildings",
    overture_type="building",
    mvt_layer_name="buildings",
    columns=(
        Column(name="id", type="string", sql_expr="id"),
        Column(name="geom", type="geometry", sql_expr="geometry"),
        Column(name="height", type="float", sql_expr="height", filterable=True),
        Column(name="num_floors", type="int", sql_expr="num_floors", filterable=True),
        Column(name="class", type="string", sql_expr="class", filterable=True),
        Column(
            name="name",
            type="string",
            sql_expr="names.primary",
            filterable=True,
        ),
    ),
    filter_aliases={
        "高さ": "height",
        "階数": "num_floors",
        "クラス": "class",
        "種別": "class",
        "名前": "name",
        "名称": "name",
    },
)
