"""Reference conversions of the frontier golden cases.

The whole-program references live in the case files (``reference_pyspark``) - they are the
oracle for synthetic inputs.  For the hybrid strategy three cases need a narrower cell,
because the rules already convert part of the program: those cells are defined here.
UNIT = the hybrid cell; WHOLE = the whole-program cell.
"""
import glob
import os

from sas2dbx.validate.conformance import load_case

_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "conformance", "frontier")
WHOLE = {c.id: c.reference_pyspark for c in map(load_case, sorted(glob.glob(os.path.join(_DIR, "*.yaml"))))}

MULTISET_UNIT = '''
lab = ctx.get("lab")
first = ctx.get("stats").orderBy("_sas_seq").select("mu").limit(1).collect()
mu = float(first[0][0]) if first and first[0][0] is not None else None
df = lab.withColumn("mu", F.lit(mu).cast("double"))
df = df.withColumn("pct", S.div(F.col("aval"), F.col("mu")) * F.lit(100.0))
ctx.put("out", df, meta={"mu": {"type": "num", "length": 8}, "pct": {"type": "num", "length": 8}})
'''

EXECUTE_UNIT = '''
src = ctx.get("src")
for (g,) in ctx.get("grps").orderBy("_sas_seq").select("grp").collect():
    ctx.put(f"out_{g}", src.where(F.col("grp") == g))
'''

RTMACRO_UNIT = '''
src = ctx.get("src")
for i in range(1, int(ctx.symget_num("ngrp")) + 1):
    ctx.put(f"part{i}", src.where(F.col("grpn") == float(i)))
'''

UNIT = {**WHOLE, "FRONTIER-MULTISET-01": MULTISET_UNIT, "FRONTIER-EXECUTE-01": EXECUTE_UNIT,
        "FRONTIER-RTMACRO-01": RTMACRO_UNIT}
