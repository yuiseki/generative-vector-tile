"""Natural Earth 10m Admin-1 States & Provinces dataset.

Source: https://z.yuiseki.net/static/natural-earth/ne_10m_admin_1_states_provinces.parquet
Built from: https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip
Rebuild script: scripts/build_natural_earth_parquet.py admin1_10m

Covers ~4596 sub-national units globally (prefectures, states, provinces,
regions, departments, etc.) with name_ja for Japanese place names.
"""

from generative_vector_tile.datasets.base import Column, Dataset

NE_ADMIN1_URL = (
    "https://z.yuiseki.net/static/natural-earth/ne_10m_admin_1_states_provinces.parquet"
)

TYPE_EN_ENUM = (
    "Administrative State",
    "Autonomous Region",
    "County",
    "Department",
    "District",
    "Emirate",
    "Metropolitan department",
    "Municipality",
    "Neutral City",
    "Overseas department",
    "Prefecture",
    "Province",
    "Region",
    "Republic",
    "State",
    "Union Territory",
)

admin1 = Dataset(
    id="admin1",
    description=(
        "Natural Earth 10m states & provinces — sub-national administrative units "
        "(prefectures, states, provinces, regions, departments, etc.)."
    ),
    parquet_urls=(NE_ADMIN1_URL,),
    use_st_intersects=True,
    mvt_layer_name="admin1",
    columns=(
        Column(name="geom", type="geometry", sql_expr="geometry"),
        Column(name="name", type="string", sql_expr="name", filterable=True),
        Column(name="name_en", type="string", sql_expr="name_en"),
        Column(name="name_ja", type="string", sql_expr="name_ja"),
        Column(name="name_alt", type="string", sql_expr="name_alt"),
        Column(
            name="admin",
            type="string",
            sql_expr="admin",
            filterable=True,
        ),
        Column(
            name="iso_a2",
            type="string",
            sql_expr="iso_a2",
            filterable=True,
        ),
        Column(
            name="iso_3166_2",
            type="string",
            sql_expr="iso_3166_2",
            filterable=True,
        ),
        Column(
            name="adm0_a3",
            type="string",
            sql_expr="adm0_a3",
            filterable=True,
        ),
        Column(
            name="type_en",
            type="string",
            sql_expr="type_en",
            filterable=True,
            enum_values=TYPE_EN_ENUM,
        ),
        Column(
            name="region",
            type="string",
            sql_expr="region",
            filterable=True,
        ),
        Column(name="area_sqkm", type="float", sql_expr="area_sqkm", filterable=True),
    ),
    filter_aliases={
        "名前": "name",
        "名称": "name",
        "国名": "admin",
        "国": "admin",
        "国コード": "iso_a2",
        "ISO": "iso_3166_2",
        "種別": "type_en",
        "地域": "region",
        "面積": "area_sqkm",
        # 行政区画タイプ
        "都道府県": "type_en",
        "prefecture": "type_en",
        "Prefecture": "type_en",
        "州": "type_en",
        "state": "type_en",
        "State": "type_en",
        "県": "type_en",
        "province": "type_en",
        "Province": "type_en",
        "地区": "type_en",
        "district": "type_en",
        "Department": "type_en",
        # 国名
        "日本": "admin",
        "アメリカ": "admin",
        "中国": "admin",
        "インド": "admin",
        "ブラジル": "admin",
        "ドイツ": "admin",
        "フランス": "admin",
    },
)
