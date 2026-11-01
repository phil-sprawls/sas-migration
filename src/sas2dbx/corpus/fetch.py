"""Download the public corpora listed in manifest.yaml at pinned commits (with SHA-256 manifest)."""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import urllib.request

import yaml

MANIFEST = os.path.join(os.path.dirname(__file__), "manifest.yaml")
RAW = "https://raw.githubusercontent.com/{repo}/{commit}/{path}"
TREE = "https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "sas2dbx-corpus-fetch"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


# spark-sas7bdat 3.0.0 (Scala 2.12, Spark 3.x) and its runtime dependencies.  On Databricks classic compute install
# the Maven library saurfang:spark-sas7bdat:3.0.0-s_2.12 from https://repos.spark-packages.org/ instead.
# parso and log4j-api-scala match Maven Central's published SHA-1; spark-packages publishes no checksum for
# spark-sas7bdat, so its SHA-256 is pinned from the official repository (HTTPS).
SPARK_SAS7BDAT_JARS = {
    "spark-sas7bdat-3.0.0-s_2.12.jar": (
        "https://repos.spark-packages.org/saurfang/spark-sas7bdat/3.0.0-s_2.12/spark-sas7bdat-3.0.0-s_2.12.jar",
        "7b2ed0add13956fdbf6c689532dbf890a11ec764529b85f35a523ca9434b483c"),
    "parso-2.0.11.jar": (
        "https://repo1.maven.org/maven2/com/epam/parso/2.0.11/parso-2.0.11.jar",
        "c3042420664fccf8634f77d99bd75e1d2ec03af985e1bf9f1c7a9f4cc79c8fe8"),
    "log4j-api-scala_2.12-12.0.jar": (
        "https://repo1.maven.org/maven2/org/apache/logging/log4j/log4j-api-scala_2.12/12.0/log4j-api-scala_2.12-12.0.jar",
        "4211e119d1b1225f86a7ab3311c9f398cf8608f37b987047406e5a47b8c152c8"),
}


def fetch_jars(dest: str = os.path.join(".tools", "jars"), log=print) -> list[str]:
    """Download the spark-sas7bdat jars; any SHA-256 mismatch deletes the file and stops (never load unchecked code)."""
    os.makedirs(dest, exist_ok=True)
    paths = []
    for name, (url, sha256) in SPARK_SAS7BDAT_JARS.items():
        path = os.path.join(dest, name)
        if not os.path.exists(path):
            with open(path, "wb") as fh:
                fh.write(_get(url))
        with open(path, "rb") as fh:
            got = hashlib.sha256(fh.read()).hexdigest()
        if got != sha256:
            os.remove(path)
            raise RuntimeError(f"{name}: SHA-256 {got} does not match the pinned {sha256}")
        log(f"{name}: SHA-256 verified")
        paths.append(path)
    return paths


def load_manifest() -> dict:
    with open(MANIFEST) as fh:
        return yaml.safe_load(fh)


def _derive(root: str, local: str, recipe: dict) -> str:
    """A golden-set variant built from a fetched file: only the recipe lives in the repository, so third-party code
    without a licence is never redistributed.  Every replacement must apply; the result must match its SHA-256."""
    with open(os.path.join(root, recipe["from"]), encoding="utf-8") as fh:
        text = fh.read()
    for old, new in recipe.get("replace", []):
        if old not in text:
            raise ValueError(f"{local}: {old!r} not found in {recipe['from']}")
        text = text.replace(old, new)
    data = (recipe.get("header", "") + text).encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    if recipe.get("sha256") and digest != recipe["sha256"]:
        raise ValueError(f"{local}: derived file has SHA-256 {digest}, expected {recipe['sha256']}")
    path = os.path.join(root, local)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)
    return digest


def fetch(dest: str = "corpus", only: list[str] | None = None, log=print) -> dict:
    man = load_manifest()
    record: dict = {}
    for name, src in man["sources"].items():
        if only and name not in only:
            continue
        root = os.path.join(dest, name)
        os.makedirs(root, exist_ok=True)
        targets: dict[str, str] = dict(src.get("files", {}))
        if src.get("globs"):
            tree = json.loads(_get(TREE.format(repo=src["repo"], commit=src["commit"])))
            blobs = [t["path"] for t in tree.get("tree", []) if t["type"] == "blob"]
            for sub, pattern in src["globs"].items():
                for p in blobs:
                    if fnmatch.fnmatch(p, pattern):
                        local = os.path.join(sub, os.path.basename(p)) if "**" not in pattern else os.path.join(sub, p)
                        targets[local] = p
        files = {}
        for local, remote in sorted(targets.items()):
            path = os.path.join(root, local)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if not os.path.exists(path):
                data = _get(RAW.format(repo=src["repo"], commit=src["commit"], path=urllib.request.quote(remote)))
                with open(path, "wb") as fh:
                    fh.write(data)
            with open(path, "rb") as fh:
                files[local] = hashlib.sha256(fh.read()).hexdigest()
        for local, recipe in sorted((src.get("derived") or {}).items()):
            files[local] = _derive(root, local, recipe)
        record[name] = {"repo": src["repo"], "commit": src["commit"], "license": src.get("license"), "files": files}
        log(f"{name}: {len(files)} files @ {src['commit'][:10]}")
        with open(os.path.join(root, "MANIFEST.json"), "w") as fh:
            json.dump(record[name], fh, indent=2, sort_keys=True)
    return record
