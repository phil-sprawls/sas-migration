"""Runtime library imported by generated notebooks: ``S`` (functions), ``P`` (steps), ``SasContext``.

Submodules load lazily so the converter (which only needs ``pyfuncs``/``dispatch``)
never requires a Spark session.
"""
from __future__ import annotations

import os


def __getattr__(name: str):
    if name == "S":
        from . import functions
        return functions
    if name == "P":
        from . import procs
        return procs
    if name == "SasContext":
        from .context import SasContext
        return SasContext
    raise AttributeError(name)


def context_from_widgets(spark, default_config: str = "sas2dbx.yaml", params=()):
    """Build the context from a Databricks widget/job parameter ``sas2dbx_config`` (YAML path).

    ``params`` are the program's run-time parameters (macro variables the SAS program got from outside:
    prompts, a parameter module, -SYSPARM).  Each is read from the job/notebook parameter of the same name;
    one that is not supplied stays undefined, so the run fails where it is first used instead of using a blank.
    """
    from .context import SasContext

    path = None
    widgets = None
    try:  # pragma: no cover - only on Databricks
        from pyspark.dbutils import DBUtils

        widgets = DBUtils(spark).widgets
        try:
            path = widgets.get("sas2dbx_config")
        except Exception:
            path = None
    except Exception:
        path = None
    path = path or os.environ.get("SAS2DBX_CONFIG") or default_config
    ctx = SasContext.from_config(spark, path) if os.path.exists(path) else SasContext(spark)
    for name in params:
        value = None
        if widgets is not None:  # pragma: no cover - only on Databricks
            try:
                value = widgets.get(name)
            except Exception:
                try:
                    widgets.text(name, "")         # interactive runs: show the parameter as a notebook widget
                except Exception:
                    pass
        if value not in (None, "") or name.upper() == "SYSPARM":
            ctx.symput(name, value or "")
    return ctx


__all__ = ["S", "P", "SasContext", "context_from_widgets"]
