from generative_vector_tile.datasets.base import Column, Dataset

OVERTURE_RELEASE = "2026-04-15.0"

# Overture Transportation `segment` schema (release 2026-04-15.0).
# Reference:
# https://docs.overturemaps.org/schema/reference/transportation/segment/

# Top-level subtype distinguishes road / rail / water.
TRANSPORTATION_SUBTYPE_ENUM = (
    "road",
    "rail",
    "water",
)

# Class enum: depends on subtype but combined here so the LLM can map any
# Japanese term ("高速道路", "新幹線", "地下鉄") to the right value without
# first having to guess the subtype.
TRANSPORTATION_CLASS_ENUM = (
    # subtype=road
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "unclassified", "residential", "living_street", "service",
    "pedestrian", "footway", "cycleway", "path", "bridleway", "steps",
    "track", "busway", "raceway", "parking_aisle", "driveway",
    # subtype=rail
    "rail", "subway", "light_rail", "tram", "monorail", "funicular",
    "narrow_gauge", "miniature", "preserved",
    # subtype=water (rare, mostly rivers/canals end up here)
    "river", "stream", "canal", "drain", "ditch",
)

transportation = Dataset(
    id="transportation",
    description="Overture Transportation segments — roads, rail, and water lines.",
    overture_release=OVERTURE_RELEASE,
    overture_theme="transportation",
    overture_type="segment",
    mvt_layer_name="transportation",
    columns=(
        Column(name="id", type="string", sql_expr="id"),
        Column(name="geom", type="geometry", sql_expr="geometry"),
        Column(
            name="subtype",
            type="string",
            sql_expr="subtype",
            filterable=True,
            enum_values=TRANSPORTATION_SUBTYPE_ENUM,
        ),
        Column(
            name="class",
            type="string",
            sql_expr="class",
            filterable=True,
            enum_values=TRANSPORTATION_CLASS_ENUM,
        ),
        Column(
            name="name",
            type="string",
            sql_expr="names.primary",
            filterable=True,
        ),
    ),
    filter_aliases={
        # Broad subtype
        "道路": "subtype",
        "線路": "subtype",
        "鉄道": "subtype",
        "水路": "subtype",
        "種別": "subtype",
        # Granular class
        "高速道路": "class",
        "幹線道路": "class",
        "地下鉄": "class",
        "路面電車": "class",
        "歩道": "class",
        "自転車道": "class",
        "クラス": "class",
        # Name match
        "名前": "name",
        "名称": "name",
    },
)
