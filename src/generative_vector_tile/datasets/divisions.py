"""Overture Maps Divisions dataset.

Covers administrative boundaries at all levels globally:
  country (admin_level=0), region/state/province (1), county (2-3),
  locality/city, neighborhood, etc.

Overture release: resolved to the catalog's latest at runtime.
Theme: divisions / Type: division_area
"""

from generative_vector_tile.datasets.base import Column, Dataset
from generative_vector_tile.stac_index import LATEST_RELEASE

OVERTURE_RELEASE = LATEST_RELEASE

SUBTYPE_ENUM = (
    "country",
    "dependency",
    "region",
    "county",
    "localadmin",
    "locality",
    "macrohood",
    "neighborhood",
    "microhood",
)

divisions = Dataset(
    id="divisions",
    description=(
        "Overture Divisions — administrative boundaries at all levels "
        "(country, region/state/province, county, locality/city, neighborhood)."
    ),
    overture_release=OVERTURE_RELEASE,
    overture_theme="divisions",
    overture_type="division_area",
    mvt_layer_name="divisions",
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
            name="subtype",
            type="string",
            sql_expr="subtype",
            filterable=True,
            enum_values=SUBTYPE_ENUM,
        ),
        Column(
            name="admin_level",
            type="int",
            sql_expr="admin_level",
            filterable=True,
        ),
        Column(
            name="country",
            type="string",
            sql_expr="country",
            filterable=True,
        ),
        Column(
            name="region",
            type="string",
            sql_expr="region",
            filterable=True,
        ),
        Column(
            name="class",
            type="string",
            sql_expr="class",
        ),
    ),
    filter_aliases={
        "名前": "name",
        "名称": "name",
        "種別": "subtype",
        "行政レベル": "admin_level",
        "国コード": "country",
        "国": "country",
        "地域": "region",
        # subtype shortcuts
        "国境": "subtype",
        "州": "subtype",
        "県": "subtype",
        "都道府県": "subtype",
        "郡": "subtype",
        "市": "subtype",
        "都市": "subtype",
        "地区": "subtype",
        "近隣": "subtype",
    },
)
