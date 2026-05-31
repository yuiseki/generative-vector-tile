"""Natural Earth 110m Admin-0 Countries datasets.

Source: https://z.yuiseki.net/static/natural-earth/ne_110m_admin_0_countries.parquet
Built from: https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip
Rebuild script: scripts/build_natural_earth_parquet.py

Two dataset views over the same file:
  population  -- filterable by pop_est, pop_rank, subregion, continent
  economics   -- filterable by gdp_md, economy, income_grp, region_wb
"""

from generative_vector_tile.datasets.base import Column, Dataset

NE_PARQUET_URL = (
    "https://z.yuiseki.net/static/natural-earth/ne_110m_admin_0_countries.parquet"
)

# Closed enums keep the LLM from hallucinating non-existent strings.
ECONOMY_ENUM = (
    "1. Developed region: G7",
    "2. Developed region: nonG7",
    "3. Emerging region: BRIC",
    "4. Emerging region: MIKT",
    "5. Emerging region: G20",
    "6. Developing region",
    "7. Least developed region",
)

INCOME_GRP_ENUM = (
    "1. High income: OECD",
    "2. High income: nonOECD",
    "3. Upper middle income",
    "4. Lower middle income",
    "5. Low income",
)

CONTINENT_ENUM = (
    "Africa",
    "Antarctica",
    "Asia",
    "Europe",
    "North America",
    "Oceania",
    "Seven seas (open ocean)",
    "South America",
)

REGION_WB_ENUM = (
    "East Asia & Pacific",
    "Europe & Central Asia",
    "Latin America & Caribbean",
    "Middle East & North Africa",
    "North America",
    "South Asia",
    "Sub-Saharan Africa",
    "Antarctica",
)

SUBREGION_ENUM = (
    "Australia and New Zealand",
    "Caribbean",
    "Central America",
    "Central Asia",
    "Eastern Africa",
    "Eastern Asia",
    "Eastern Europe",
    "Melanesia",
    "Micronesia",
    "Middle Africa",
    "Northern Africa",
    "Northern America",
    "Northern Europe",
    "Polynesia",
    "Seven seas (open ocean)",
    "South America",
    "South-Eastern Asia",
    "Southern Africa",
    "Southern Asia",
    "Southern Europe",
    "Western Africa",
    "Western Asia",
    "Western Europe",
    "Antarctica",
)

# Columns common to both datasets
_COMMON_COLUMNS = (
    Column(name="geom", type="geometry", sql_expr="geometry"),
    Column(name="name", type="string", sql_expr="name"),
    Column(name="name_en", type="string", sql_expr="name_en"),
    Column(name="name_ja", type="string", sql_expr="name_ja"),
    Column(name="iso_a3", type="string", sql_expr="iso_a3"),
    Column(name="iso_a2", type="string", sql_expr="iso_a2"),
    Column(name="sovereignt", type="string", sql_expr="sovereignt"),
    Column(name="type", type="string", sql_expr="type"),
)

population = Dataset(
    id="population",
    description=(
        "Natural Earth 110m countries — population statistics. "
        "Covers all ~177 sovereign states and territories globally."
    ),
    parquet_urls=(NE_PARQUET_URL,),
    use_st_intersects=True,
    mvt_layer_name="countries",
    columns=_COMMON_COLUMNS + (
        Column(
            name="pop_est",
            type="int",
            sql_expr="pop_est",
            filterable=True,
        ),
        Column(
            name="pop_rank",
            type="int",
            sql_expr="pop_rank",
            filterable=True,
        ),
        Column(
            name="continent",
            type="string",
            sql_expr="continent",
            filterable=True,
            enum_values=CONTINENT_ENUM,
        ),
        Column(
            name="subregion",
            type="string",
            sql_expr="subregion",
            filterable=True,
            enum_values=SUBREGION_ENUM,
        ),
        Column(
            name="region_wb",
            type="string",
            sql_expr="region_wb",
            filterable=True,
            enum_values=REGION_WB_ENUM,
        ),
    ),
    filter_aliases={
        "人口": "pop_est",
        "人口推計": "pop_est",
        "人口ランク": "pop_rank",
        "大陸": "continent",
        "地域": "subregion",
        "サブ地域": "subregion",
        "世界銀行地域": "region_wb",
        "アジア": "continent",
        "アフリカ": "continent",
        "ヨーロッパ": "continent",
        "北アメリカ": "continent",
        "南アメリカ": "continent",
        "オセアニア": "continent",
    },
)

economics = Dataset(
    id="economics",
    description=(
        "Natural Earth 110m countries — economic statistics (GDP, income group, economy type). "
        "Covers all ~177 sovereign states and territories globally."
    ),
    parquet_urls=(NE_PARQUET_URL,),
    use_st_intersects=True,
    mvt_layer_name="countries",
    columns=_COMMON_COLUMNS + (
        Column(
            name="gdp_md",
            type="float",
            sql_expr="gdp_md",
            filterable=True,
        ),
        Column(
            name="economy",
            type="string",
            sql_expr="economy",
            filterable=True,
            enum_values=ECONOMY_ENUM,
        ),
        Column(
            name="income_grp",
            type="string",
            sql_expr="income_grp",
            filterable=True,
            enum_values=INCOME_GRP_ENUM,
        ),
        Column(
            name="continent",
            type="string",
            sql_expr="continent",
            filterable=True,
            enum_values=CONTINENT_ENUM,
        ),
        Column(
            name="region_wb",
            type="string",
            sql_expr="region_wb",
            filterable=True,
            enum_values=REGION_WB_ENUM,
        ),
    ),
    filter_aliases={
        "GDP": "gdp_md",
        "国内総生産": "gdp_md",
        "経済規模": "gdp_md",
        "所得水準": "income_grp",
        "所得グループ": "income_grp",
        "高所得": "income_grp",
        "低所得": "income_grp",
        "経済区分": "economy",
        "途上国": "economy",
        "先進国": "economy",
        "G7": "economy",
        "BRIC": "economy",
        "大陸": "continent",
        "地域": "region_wb",
    },
)
