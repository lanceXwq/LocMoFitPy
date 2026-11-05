"""LocMoFit: a geometric model fitted to the localizations of a site, as a SMAPpy plugin.

Part of LocMoFitPy, outside SMAPpy's package: SMAPpy scans this folder (`locmofitpy/smappy_plugin`) once it is one of its plugin roots, which `python -m locmofitpy.smappy_setup` makes it. The fit is LocMoFitPy's `run_locmofit`; this file only maps SMAPpy's table and region onto it and the result back onto region values.

**Updates.** LocMoFitPy, this file with it, comes from the head of its GitHub repository's main branch: before the first fit of each SMAPpy session unless SMAPpy's automatic update check is off (*Preferences > General*); with it off, LocMoFitPy is updated by installing it into SMAPpy's Python again, as at the start. The new version is installed before anything of LocMoFitPy is imported, so the fit runs on it at once. An editable install (a checkout) is never updated here. This file reads only the standard library and SMAPpy until the fit, because what it imports of LocMoFitPy would stay the old version until SMAPpy restarts.

**Per region.** `per_region = True`: it runs on the view, once per ROI, or once per cluster, and the fitted values become columns of the region table, so *Filter ROIs*, *Sort ROIs*, histograms and *Regions to column* use them.

**The data.** Positions are `x_nm`, `y_nm` and, with `z_err_nm`, `z_nm` (a 3D fit; otherwise 2D, from the first two coordinates of the model). The precisions are `x_err_nm` and `y_err_nm` where the table has them, else `xy_err_nm` for both. The table is the one the layer shows: blinks on a grouped layer, localizations on an ungrouped one.

**The field of view.** The background is uniform over the field of view, so its volume enters the likelihood. In an ROI it is the ROI's area, times in 3D a depth: the ROI's size, as SMAP's LocMoFit takes its height, or *depth*; on the view, without an ROI, LocMoFitPy's default, the localizations' bounding box.

**Starting values.** The model's defaults, with the position at the ROI's centre (z at the localizations' median), and whatever *start values* name.

**Values.** Each component in the world frame: its position `x`, `y`, `z`, axis angles `theta`, `phi` (radians) and shape (`r`; or `s`, `kappa`, `curvature`, `radius`, `alpha`, `area`), with the component's number appended for a composite (`r1`, `r2`); then `sigma`, `signal_fraction`, the components' shares `w1`, `w2`, …, `loss`, `iterations` and `converged`.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import math
import subprocess
import urllib.request
from dataclasses import dataclass
from importlib import metadata

import numpy as np
from smappy import updates
from smappy.plugins import Context, Plot, Plugin, Result, param, register

MODELS = ["Ring", "SphericalCap", "DoubleRing", "SpCapRing"]
FIGURE_POINTS = 3000

DISTRIBUTION = "LocMoFitPy"
REPOSITORY = "https://github.com/lanceXwq/LocMoFitPy"
BRANCH = "main"
HEAD = f"https://api.github.com/repos/lanceXwq/LocMoFitPy/commits/{BRANCH}"
MISSING = f"LocMoFit needs LocMoFitPy: install it into SMAPpy's Python from git+{REPOSITORY}"
_asked = False  # whether this session has asked GitHub yet


def installed_source() -> tuple[bool, str | None, bool]:
    """Whether LocMoFitPy is installed, the commit it was installed from (None when not from git), and whether it is an editable install, from its package metadata without importing it."""
    try:
        text = metadata.distribution(DISTRIBUTION).read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return False, None, False
    info = json.loads(text) if text else {}
    commit = info.get("vcs_info", {}).get("commit_id")
    return True, commit, bool(info.get("dir_info", {}).get("editable"))


def newest_commit(timeout: float = 5.0) -> str:
    """The commit at the head of LocMoFitPy's main branch on GitHub."""
    request = urllib.request.Request(
        HEAD,
        headers={
            "Accept": "application/vnd.github.sha",
            "User-Agent": "smappy-locmofit",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode().strip()


def update_locmofitpy(report) -> str:
    """Install the head of LocMoFitPy's main branch when LocMoFitPy is missing or installed from another commit, with SMAPpy's own installer (this Python's pip, else uv's pip aimed at it); what was done, in a sentence. An editable install is left alone."""
    installed, commit, editable = installed_source()
    if editable:
        return "LocMoFitPy is an editable install (a checkout), so it is not updated here: update the checkout with git pull"
    newest = newest_commit()
    if installed and commit == newest:
        return f"LocMoFitPy is up to date ({newest[:7]})"
    from smappy.torch_install import installer

    report(f"installing LocMoFitPy {newest[:7]} from GitHub")
    done = subprocess.run(
        [
            *installer(),
            "--upgrade",
            f"{DISTRIBUTION} @ git+{REPOSITORY}@{newest}",
        ],
        capture_output=True,
        text=True,
    )
    if done.returncode != 0:
        raise RuntimeError(
            f"installing LocMoFitPy failed:\n{done.stderr or done.stdout}"
        )
    importlib.invalidate_caches()
    return f"LocMoFitPy {newest[:7]} installed"


def parse_values(text: str) -> dict[str, float]:
    """`"r=50, theta=0.3"` as `{"r": 50.0, "theta": 0.3}`."""
    out = {}
    for item in filter(
        None, (s.strip() for s in text.replace(";", ",").split(","))
    ):
        name, sep, value = item.partition("=")
        if not sep:
            raise ValueError(
                f"start values are name=value pairs, not {item!r}"
            )
        out[name.strip()] = float(value)
    return out


def parse_names(text: str) -> tuple[str, ...]:
    return tuple(
        filter(None, (s.strip() for s in text.replace(";", ",").split(",")))
    )


def parse_ties(text: str) -> dict[str, str]:
    """`"r2=r1"` as `{"r2": "r1"}`: the target takes the source's value."""
    out = {}
    for item in parse_names(text):
        target, sep, source = item.partition("=")
        if not sep:
            raise ValueError(f"ties are target=source pairs, not {item!r}")
        out[target.strip()] = source.strip()
    return out


def site_arrays(locs) -> tuple[np.ndarray, np.ndarray]:
    """The (N, D) positions and per-axis precisions (nm) of a table: 3D when it has `z_nm` and `z_err_nm`."""
    missing = [c for c in ("x_nm", "y_nm") if c not in locs]
    if missing:
        raise ValueError(
            f"the table has no {', '.join(missing)}; it has "
            + ", ".join(sorted(locs.keys()))
        )
    if "x_err_nm" in locs and "y_err_nm" in locs:
        lateral = [locs["x_err_nm"], locs["y_err_nm"]]
    elif "xy_err_nm" in locs:
        lateral = [locs["xy_err_nm"]] * 2
    else:
        raise ValueError(
            "the table has no precision (xy_err_nm, or x_err_nm and y_err_nm); it has "
            + ", ".join(sorted(locs.keys()))
        )
    columns, errors = [locs["x_nm"], locs["y_nm"]], lateral
    if "z_nm" in locs and "z_err_nm" in locs:
        columns.append(locs["z_nm"])
        errors.append(locs["z_err_nm"])
    positions = np.column_stack([np.asarray(c, dtype=float) for c in columns])
    precisions = np.column_stack([np.asarray(c, dtype=float) for c in errors])
    keep = np.all(np.isfinite(positions), axis=1) & np.all(
        np.isfinite(precisions) & (precisions > 0), axis=1
    )
    return positions[keep], precisions[keep]


def region_area(region) -> float:
    """The area of an ROI (nm²): a circle's π r², else the shoelace formula on its outline."""
    if region.kind == "circle":
        return math.pi * region.radius**2
    v = (
        region.vertices()
        if region.kind == "rect"
        else np.asarray(region.points, dtype=float)
    )
    x, y = v[:, 0], v[:, 1]
    return float(
        abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2
    )


def field_of_view(
    region, positions: np.ndarray, depth: float = 0.0
) -> float | None:
    """The background's field of view: the ROI's area, times in 3D a depth -- `depth` (nm), or 0 for the ROI's size (a circle's diameter, else the square root of its area), the height SMAP's LocMoFit gives its cylinder. The localizations' own z span is a poor estimate: their axial precision is heavy-tailed, so it overestimates the depth (a median of 419 nm on LocMoFit.jl's benchmark ring sites, whose background fills 300 nm). None without an ROI (LocMoFitPy then takes the localizations' bounding box)."""
    if region is None:
        return None
    area = region_area(region)
    if positions.shape[1] == 2:
        return area
    size = 2 * region.radius if region.kind == "circle" else math.sqrt(area)
    return area * (depth if depth > 0 else size)


def start_values(
    model: str, region, positions: np.ndarray, given: dict[str, float]
) -> dict[str, float]:
    """The position at the ROI's centre (z at the localizations' median), under the model's own name for it (`x` or `x1`), then what the user gave."""
    from locmofitpy import get_model

    names = get_model(model).default_parameters()
    keys = ("x", "y", "z") if "x" in names else ("x1", "y1", "z1")
    center = (
        region.center
        if region is not None
        else np.median(positions[:, :2], axis=0)
    )
    start = {keys[0]: float(center[0]), keys[1]: float(center[1])}
    if positions.shape[1] == 3:
        start[keys[2]] = float(np.median(positions[:, 2]))
    return {**start, **given}


def flat_values(result: dict) -> dict[str, float]:
    """A fit's result as one number per name: each component in the world frame, then the shared values."""
    components = result["components"]
    out = {}
    for k, c in enumerate(components, start=1):
        suffix = "" if len(components) == 1 else str(k)
        for axis, value in zip("xyz", c["position"]):
            out[f"{axis}{suffix}"] = float(value)
        for name, value in c.items():
            if name not in ("position", "axis"):
                out[f"{name}{suffix}"] = float(value)
    # the likelihood depends on sigma only through sigma²
    out["sigma"] = abs(float(result["parameters"]["sigma"]))
    fraction = result["signal_fraction"]
    out["signal_fraction"] = (
        float("nan") if fraction is None else float(fraction)
    )
    if len(components) > 1:
        out.update(
            {
                f"w{k}": float(w)
                for k, w in enumerate(result["weights"], start=1)
            }
        )
    out["loss"] = float(result["loss"])
    out["iterations"] = float(result["iterations"])
    out["converged"] = float(result["retcode"] == "Success")
    return out


def draw_view(
    ax, positions: np.ndarray, points: np.ndarray, i: int, j: int
) -> None:
    """The localizations (grey) and the fitted model's points (red), along axes i and j."""
    step = max(1, len(positions) // FIGURE_POINTS)
    pstep = max(1, len(points) // FIGURE_POINTS)
    ax.plot(
        positions[::step, i],
        positions[::step, j],
        ".",
        color="0.5",
        markersize=3,
        alpha=0.6,
    )
    ax.plot(
        points[::pstep, i],
        points[::pstep, j],
        ".",
        color="#e0453a",
        markersize=1.5,
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("xyz"[i] + " (nm)")
    ax.set_ylabel("xyz"[j] + " (nm)")


def figure(positions: np.ndarray, points: np.ndarray) -> Plot:
    """xy, and in 3D also xz beside it (SMAPpy hands one axis to a one-panel plot, the figure to more)."""
    if positions.shape[1] == 2:
        return Plot(
            lambda ax: draw_view(ax, positions, points, 0, 1), size=(4.2, 4.0)
        )

    def draw(fig):
        a, b = fig.subplots(1, 2)
        draw_view(a, positions, points, 0, 1)
        draw_view(b, positions, points, 0, 2)

    return Plot(draw, panels=2, size=(8.4, 4.0))


@dataclass
class LocMoFitSettings:
    model: str = param(
        "Ring",
        label="model",
        choices=MODELS,
        help="the geometric model fitted to each site",
    )
    start: str = param(
        "",
        label="start values",
        kind="text",
        help="name=value, ...: replaces the defaults; the position starts at the ROI's centre",
    )
    freeze: str = param(
        "",
        label="freeze",
        kind="text",
        help="parameters kept at their start values, e.g. vx, vy",
    )
    tie: str = param(
        "",
        label="tie",
        kind="text",
        advanced=True,
        help="target=source, ...: the target takes the source's value, e.g. r2=r1",
    )
    spacing_nm: float = param(
        3.0,
        label="point spacing",
        unit="nm",
        min=0.1,
        help="the distance between the model's points",
    )
    background: bool = param(
        True,
        label="background",
        help="some localizations are background, uniform over the field of view",
    )
    depth_nm: float = param(
        0.0,
        label="depth",
        unit="nm",
        min=0.0,
        help="the field of view's depth in 3D; 0: the ROI's size",
    )
    dtype: str = param(
        "float64",
        label="precision",
        choices=["float64", "float32"],
        advanced=True,
        help="float32 is faster but can stop early",
    )
    max_iter: int = param(
        200,
        label="iterations",
        min=1,
        advanced=True,
        help="the most BFGS iterations",
    )
    tol: float = param(
        1e-4,
        label="tolerance",
        min=0.0,
        advanced=True,
        help="converged when no gradient component exceeds this",
    )


@register("Analysis/Measure/LocMoFit")
class LocMoFit(Plugin):
    """A geometric model (ring, cap, double ring, cap with ring) fitted to the localizations by maximum likelihood, per ROI or cluster."""

    Settings = LocMoFitSettings
    changes_locs = False
    per_region = True
    version = "1"

    def run(self, ctx: Context, settings: LocMoFitSettings) -> Result:
        global _asked
        if not _asked and updates.automatic():
            _asked = True
            try:
                ctx.report(update_locmofitpy(ctx.report))
            except (OSError, RuntimeError) as error:
                # offline, GitHub slow, or the install failed: fit with what is installed
                ctx.report(f"LocMoFitPy was not updated: {error}")
        if importlib.util.find_spec("locmofitpy") is None:
            raise ValueError(MISSING)
        from locmofitpy import run_locmofit

        positions, precisions = site_arrays(ctx.selection.apply(ctx.locs))
        if len(positions) < 10:
            raise ValueError(
                f"{len(positions)} localizations, too few for a fit"
            )
        region = ctx.region
        init = start_values(
            settings.model, region, positions, parse_values(settings.start)
        )
        ctx.report(
            f"fitting {settings.model} to {len(positions)} localizations"
        )
        result = run_locmofit(
            settings.model,
            positions,
            precisions,
            init_params=init,
            freeze=parse_names(settings.freeze),
            tie=parse_ties(settings.tie) or None,
            max_iter=settings.max_iter,
            tol=settings.tol,
            spacing=settings.spacing_nm,
            dtype=settings.dtype,
            background=settings.background,
            fov_volume=field_of_view(region, positions, settings.depth_nm),
        )
        values = flat_values(result)
        points = np.asarray(result["model_points"])[:, : positions.shape[1]]
        text = (
            f"{settings.model}, {len(positions)} localizations in {positions.shape[1]}D: {result['retcode']} after {result['iterations']} iterations; "
            + ", ".join(
                f"{k} {v:.4g}"
                for k, v in values.items()
                if k not in ("loss", "iterations", "converged")
            )
        )
        return Result(
            text=text,
            plot=figure(positions, points),
            values=values,
            data={"result": result},
            settings=settings,
            title=f"LocMoFit: {settings.model}",
        )
