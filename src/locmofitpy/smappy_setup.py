"""Make LocMoFitPy's plugin for SMAPpy, *LocMoFit*, known to SMAPpy: run `python -m locmofitpy.smappy_setup` with SMAPpy's Python, once, after installing LocMoFitPy into it (`pip install "LocMoFitPy @ git+https://github.com/lanceXwq/LocMoFitPy"`, or `uv pip install --python <SMAPpy's python> ...` where it has no pip). It adds the folder of plugins this package ships, `smappy_plugin`, to SMAPpy's plugin roots; from then on the plugin keeps LocMoFitPy up to date from GitHub."""

from pathlib import Path


def main() -> None:
    from smappy import config

    root = Path(__file__).resolve().parent / "smappy_plugin"
    roots = [Path(r) for r in config.plugin_roots()]
    if root not in roots:
        config.set_plugin_roots([*map(str, roots), str(root)])
    print(f"SMAPpy finds LocMoFit in {root}")


if __name__ == "__main__":
    main()
