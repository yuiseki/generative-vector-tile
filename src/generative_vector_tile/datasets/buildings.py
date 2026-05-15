from generative_vector_tile.datasets.base import Column, Dataset

OVERTURE_RELEASE = "2026-04-15.0"

# Overture buildings schema (release 2026-04-15.0) actually puts the broad
# rollup in `subtype` and the fine-grained value in `class`, opposite to
# what the column names suggest. Verified empirically against the Tokyo
# bbox: `class` contained values like 'school', 'hotel', 'stadium',
# 'temple', 'apartments' while `subtype` contained 'residential',
# 'commercial', 'education', 'religious'. We list the enums to match
# reality so the LLM emits values that actually exist in the data.
#
# Note: only ~4% of Tokyo buildings have non-null class / subtype. Even
# correctly-targeted filters will return very few hits in most areas; the
# rest of the data is unclassified.

# Broad rollup. ~14 values.
BUILDINGS_SUBTYPE_ENUM = (
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

# Fine-grained category. ~50 values, rolls up to subtype.
BUILDINGS_CLASS_ENUM = (
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
    # observed in Tokyo data but not in upstream schema docs
    "roof", "public",
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
        # Granular building types are stored in `class` (post-schema-fix).
        "学校": "class",
        "病院": "class",
        "競技場": "class",
        "ホテル": "class",
        "寺": "class",
        "教会": "class",
        "工場": "class",
        "倉庫": "class",
        "アパート": "class",
        # Broad categories live in `subtype`.
        "教育施設": "subtype",
        "商業施設": "subtype",
        "居住施設": "subtype",
    },
)
