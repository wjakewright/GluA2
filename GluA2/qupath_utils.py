import pandas as pd
import json, re
import numpy as np
import pandas as pd
import tifffile as tiff
from shapely.geometry import shape, box, Polygon, MultiPolygon, mapping
from shapely.ops import unary_union
from rasterio.features import rasterize  # pip install rasterio

_LAYER_RE = re.compile(r"^(.*?)(1|2/3|5a|5b|5|6a|6b|6)$", flags=re.IGNORECASE)


# ----------- helpers -----------
def parse_area_side(props: dict):
    cls = props.get("classification") or {}
    names = cls.get("names") or []
    lr = {"left": "L", "right": "R"}
    side = next(
        (lr[str(n).strip().lower()] for n in names if str(n).strip().lower() in lr),
        None,
    )
    area = next(
        (str(n).strip() for n in names if str(n).strip().lower() not in lr), None
    )
    if area is None:
        area = props.get("name") or "unknown_area"

    return area, side


def normalize_name(area: str) -> str:
    m = _LAYER_RE.match(area.strip())
    if m:
        prefix, layer = m.groups()
        prefix = prefix.rstrip("_- ")
        return f"{prefix}_{layer}"
    return area.strip()


def fix_clip(geom_json, W, H):
    g = shape(geom_json)
    if not g.is_valid:
        g = g.buffer(0)
    inter = g.intersection(box(0, 0, W, H))
    if inter.is_empty:
        return None
    if isinstance(inter, (Polygon, MultiPolygon)):
        return inter
    polys = [p for p in getattr(inter, "geoms", []) if isinstance(p, Polygon)]
    return unary_union(polys) if polys else None


def mask_from_geom(g, H, W):
    """Rasterize a single geometry to a boolean mask (all_touched=True)."""
    return rasterize(
        [(mapping(g), 1)], out_shape=(H, W), fill=0, dtype="uint8", all_touched=True
    ).astype(bool)


def summarize_mask(mask, num, den, eps=1e-9):
    if not mask.any():
        return 0, 0.0, 0.0, np.nan, np.nan
    vN = num[mask]
    vD = den[mask]
    n = int(mask.sum())
    sN = float(vN.sum())
    sD = float(vD.sum())
    mean_ratio = float((vN / (vN + vD + eps)).mean())
    ratio_of_sums = sN / (sN + sD + eps)
    return n, sN, sD, mean_ratio, ratio_of_sums


def drop_branches(df: pd.DataFrame, superparent_label, recursive=True):
    """
    Remove ROIs that belong under superparent_label.
    If recursive=True, removes the whole subtree beneath any superparent_label parent.
    """
    d = df.copy()

    # Always drop rows whose immediate parent is superparent_label
    immediate_mask = d.get("parent_name") == superparent_label

    if not recursive:
        return d.loc[~immediate_mask].reset_index(drop=True)

    # --- recursive: drop whole subtree under any parent ---
    # Seeds: (a) any parent_id referenced where parent_name == fiber_label
    seeds = set(d.loc[immediate_mask, "parent_id"].dropna().tolist())

    if "roi_name" in d.columns and "orig_id" in d.columns:
        seeds |= set(
            d.loc[d["roi_name"] == superparent_label, "orig_id"].dropna().tolist()
        )

    # Build parent -> children map (using orig_id / parent_id graph)
    parent_to_children = (
        d.dropna(subset=["parent_id", "orig_id"])
        .groupby("parent_id")["orig_id"]
        .apply(lambda s: set(s.dropna().tolist()))
        .to_dict()
    )

    # Traverse to collect all descendants of seeds
    to_drop_ids = set()
    frontier = set(seeds)
    while frontier:
        pid = frontier.pop()
        if pid in to_drop_ids:
            continue
        to_drop_ids.add(pid)
        for child in parent_to_children.get(pid, ()):
            if child not in to_drop_ids:
                frontier.add(child)

    # Final mask: drop immediate children of Fiber tracts AND any descendant IDs
    drop_mask = immediate_mask | d["orig_id"].isin(to_drop_ids)
    return d.loc[~drop_mask].reset_index(drop=True)
