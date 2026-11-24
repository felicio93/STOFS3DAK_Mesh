"""Unit tests that need no OCSMesh and no real data."""
import numpy as np
import pytest
from shapely.geometry import box

from stofs3dak_mesh.config import ConfigError, parse
from stofs3dak_mesh.inputs import refinement_parts, shift_to_360

BASE = dict(name="t", data_dir="/data", domain="d.shp", dem="dem.tif",
            hmin=500, hmax=3500,
            refinements=[{"name": "a", "kind": "patch", "shp": "a.shp",
                          "size": 1500, "rate": 1e-5}])


def test_parse_paths_and_defaults():
    cfg = parse(BASE)
    assert str(cfg.domain) == "/data/d.shp"
    assert str(cfg.refinements[0].shp) == "/data/a.shp"
    assert cfg.method == "fast" and cfg.force_metric_crs


def test_parse_data_dir_override():
    cfg = parse(BASE, data_dir="/other")
    assert str(cfg.refinements[0].shp) == "/other/a.shp"


@pytest.mark.parametrize("change", [
    {"hmin": 4000},
    {"method": "slow"},
    {"bogus": 1},
    {"refinements": [{"name": "a", "kind": "blob", "shp": "a.shp", "size": 1}]},
    {"refinements": [{"name": "a", "kind": "patch", "shp": "a.shp"}]},
    {"refinements": [{"name": "a", "kind": "patch", "shp": "a.shp", "size": 1},
                     {"name": "a", "kind": "patch", "shp": "b.shp", "size": 1}]},
])
def test_parse_errors(change):
    with pytest.raises(ConfigError):
        parse({**BASE, **change})


def test_shift_to_360():
    g = shift_to_360([box(-170, 60, -160, 62)])[0]
    assert g.bounds == (190.0, 60.0, 200.0, 62.0)
    assert shift_to_360([box(170, 60, 175, 62)])[0].bounds == (170.0, 60.0, 175.0, 62.0)


def test_refinement_parts_size_field():
    import geopandas as gpd
    cfg = parse({**BASE, "refinements": [
        {"name": "h", "kind": "patch", "shp": "h.shp", "size": 500,
         "rate": 1e-4, "size_field": "size_m"}]})
    gdf = gpd.GeoDataFrame({"size_m": [np.nan, 800.0, 800.0]},
                           geometry=[box(190, 60, 191, 61), box(195, 60, 196, 61),
                                     box(197, 60, 198, 61)], crs="EPSG:4326")
    parts = refinement_parts(cfg.refinements[0], gdf)
    assert [(p[1], p[2]) for p in parts] == [(500, 1e-4), (800.0, 1e-4)]
    assert parts[1][0].area == pytest.approx(2.0)


def test_relative_data_dir_from_recipe_folder(tmp_path):
    cfg = parse({**BASE, "data_dir": "../inputs"}, source=tmp_path / "configs" / "r.yaml")
    assert cfg.data_dir == tmp_path / "inputs"
    assert cfg.domain == tmp_path / "inputs" / "d.shp"


def test_refinement_parts_missing_size_field_uses_default():
    import geopandas as gpd
    cfg = parse({**BASE, "refinements": [
        {"name": "h", "kind": "patch", "shp": "h.shp", "size": 750,
         "rate": 1e-4, "size_field": "size_m"}]})
    gdf = gpd.GeoDataFrame({"id": [1, 2]}, geometry=[box(190, 60, 191, 61),
                           box(195, 60, 196, 61)], crs="EPSG:4326")
    parts = refinement_parts(cfg.refinements[0], gdf)
    assert [(p[1], p[2]) for p in parts] == [(750, 1e-4)]


def test_elevation_limits(tmp_path):
    import geopandas as gpd
    from stofs3dak_mesh.hgrid import elevation_limits
    gpd.GeoDataFrame({"id": [1]}, geometry=[box(190, 60, 192, 62)],
                     crs="EPSG:4326").to_file(tmp_path / "zero.shp")
    gpd.GeoDataFrame({"id": [1]}, geometry=[box(-170, 60, -169, 62)],   # 190..191 after the shift
                     crs="EPSG:4326").to_file(tmp_path / "five.shp")
    gpd.GeoDataFrame({"Depth": [-30.0, None]},
                     geometry=[box(191.5, 60, 192.5, 62), box(195, 60, 196, 61)],
                     crs="EPSG:4326").to_file(tmp_path / "ch.shp")
    # nodes: in zero+five, in zero, zero+channel, channel only, elsewhere, no-depth channel
    x = np.array([190.5, 191.2, 191.8, 192.2, 194.0, 195.5])
    y = np.full(6, 61.0)
    z = np.array([8.0, 3.0, 2.0, -5.0, -1.0, 1.0])
    hc = {"max_elevation": -3,
          "max_elevation_areas": [{"shp": tmp_path / "zero.shp", "max_elevation": 0},
                                  {"shp": tmp_path / "five.shp", "max_elevation": 5}],
          "force_channels": {"shp": tmp_path / "ch.shp", "depth_field": "Depth"}}
    out, rep = elevation_limits(x, y, z, hc)
    assert out.tolist() == [5.0, 0.0, -30.0, -30.0, -3.0, -3.0]
    assert rep["areas"][1]["nodes_overlapping_earlier_areas"] == 1
    assert rep["channels"][1]["skipped"]
