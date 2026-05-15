from generative_vector_tile.datasets.base import Column, Dataset

OVERTURE_RELEASE = "2026-04-15.0"

# Overture Buildings v1.x `class` enum. Reference:
# https://docs.overturemaps.org/schema/reference/buildings/building/
BUILDINGS_CLASS_ENUM = (
    "agricultural",
    "civic",
    "commercial",
    "education",
    "entertainment",
    "industrial",
    "medical",
    "military",
    "outbuilding",
    "religious",
    "residential",
    "service",
    "transportation",
)

# Overture Buildings `subtype` enum (~50 specific values rolling up to `class`).
# This is the column to filter on for fine-grained categories like stadium,
# school, hospital, temple, hotel -- exposing it gives the LLM something real
# to map natural-language queries to instead of guessing class values.
BUILDINGS_SUBTYPE_ENUM = (
    # agricultural
    "agricultural", "barn", "farm_auxiliary", "greenhouse", "silo", "stable",
    # civic
    "civic", "fire_station", "police", "post_office", "town_hall",
    "government", "court_house", "library", "community_centre", "museum",
    # commercial
    "commercial", "office", "retail", "supermarket", "shop", "kiosk",
    "restaurant", "bank",
    # education
    "education", "school", "kindergarten", "college", "university",
    # entertainment
    "entertainment", "casino", "cinema", "theatre", "concert_hall",
    "stadium", "sports_centre", "sports_hall",
    # industrial
    "industrial", "factory", "warehouse", "manufacture",
    # medical
    "medical", "hospital", "clinic",
    # military
    "military",
    # outbuilding
    "outbuilding", "garage", "shed", "carport",
    # religious
    "religious", "cathedral", "chapel", "church", "mosque", "shrine",
    "synagogue", "temple",
    # residential
    "residential", "apartments", "house", "detached", "terrace",
    "bungalow", "dormitory", "hotel",
    # service
    "service",
    # transportation
    "transportation", "train_station", "parking", "hangar",
)

buildings = Dataset(
    id="buildings",
    description="Overture Buildings — building footprints with height, class, subtype.",
    overture_release=OVERTURE_RELEASE,
    overture_theme="buildings",
    overture_type="building",
    mvt_layer_name="buildings",
    columns=(
        Column(name="id", type="string", sql_expr="id"),
        Column(name="geom", type="geometry", sql_expr="geometry"),
        Column(name="height", type="float", sql_expr="height", filterable=True),
        Column(name="num_floors", type="int", sql_expr="num_floors", filterable=True),
        Column(
            name="class",
            type="string",
            sql_expr="class",
            filterable=True,
            enum_values=BUILDINGS_CLASS_ENUM,
        ),
        Column(
            name="subtype",
            type="string",
            sql_expr="subtype",
            filterable=True,
            enum_values=BUILDINGS_SUBTYPE_ENUM,
        ),
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
        "種別": "subtype",
        "サブタイプ": "subtype",
        "名前": "name",
        "名称": "name",
        # The user almost always cares about subtype-level granularity when
        # naming a building type in Japanese ("学校", "病院" etc.), so map
        # those to subtype rather than the coarser class.
        "学校": "subtype",
        "病院": "subtype",
        "競技場": "subtype",
        "ホテル": "subtype",
        "寺": "subtype",
        "教会": "subtype",
    },
)
