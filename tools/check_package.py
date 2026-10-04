#!/usr/bin/env python3
"""Structural check for Splosh's `Package.swift`, run against `swift package dump-package` output.

Authority: `IMPLEMENTATION-PLAN.md` §4.1 (the M0.1 replacement gate) and `M0-WORK-ORDER.md` M0.1.
The dependency graph is rev4 §3.1 as amended by `IMPLEMENTATION-PLAN.md` §4.4 (`SploshCLI` takes a
direct `SploshRuntime` edge in addition to `SploshServer`).

The checker asserts:

  * the deployment target is exactly `.macOS("27.0")`;
  * there is exactly one direct external dependency -- Hummingbird, required `exact: "2.27.0"`;
  * the package declares exactly the nine targets of rev4 §3.1;
  * it declares exactly one product -- an executable named `splosh`, built from `SploshCLI`.
    The product name is load-bearing: every CLI gate in the plan invokes `swift run splosh ...`
    and M0.5 asserts `test -x "$(swift build --show-bin-path)/splosh"`. With no explicit product
    SwiftPM names the executable after its target, and every one of those gates fails. Checking it
    here turns a misnamed executable into an M0.1 failure rather than an M0.5 surprise;
  * every target's dependency edges match the required graph (by-name edges and external-product
    edges both);
  * every target has at least one `.swift` source file on disk.

That last check reads the filesystem rather than the JSON, because `dump-package` emits `sources`
only when a manifest declares them explicitly -- and SwiftPM itself fails a target with no sources
(`error: Source files for target X should be located under Sources/X`). A target's directory is
`Tests/<name>` for a test target and `Sources/<name>` otherwise.

Usage:
    python3 tools/check_package.py /tmp/splosh-package.json

Exit code is 0 iff every check passes; otherwise each mismatch is named and the exit is 1.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# --- The required structure -------------------------------------------------------------

MACOS_VERSION = "27.0"

DEPENDENCY_IDENTITY = "hummingbird"
DEPENDENCY_URL = "https://github.com/hummingbird-project/hummingbird.git"
DEPENDENCY_EXACT = "2.27.0"

# product name -> (executable?, targets)
REQUIRED_PRODUCTS: dict[str, tuple[bool, list[str]]] = {
    "splosh": (True, ["SploshCLI"]),
}

# target name -> (by-name SwiftPM edges, external product names)
REQUIRED_TARGETS: dict[str, tuple[list[str], list[str]]] = {
    "SploshCore": ([], []),
    "SploshQuant": (["SploshCore"], []),
    "SploshModel": (["SploshCore", "SploshQuant"], []),
    "SploshOracle": (["SploshModel", "SploshQuant", "SploshCore"], []),
    "SploshRuntime": (["SploshModel", "SploshCore"], []),
    "SploshServer": (["SploshRuntime", "SploshModel", "SploshCore"], ["Hummingbird"]),
    "SploshCLI": (
        [
            "SploshRuntime",
            "SploshServer",
            "SploshModel",
            "SploshQuant",
            "SploshCore",
            "SploshOracle",
        ],
        [],
    ),
    "SploshOracleTests": (["SploshModel", "SploshQuant", "SploshCore", "SploshOracle"], []),
    "SploshServerTests": (["SploshServer", "SploshRuntime", "SploshCLI"], []),
}

# The package root owns the manifest; the source-tree check is relative to it, never to $PWD.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent


class Findings:
    """Collects every mismatch so one run reports all of them, not just the first."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checks = 0

    def ok(self, label: str) -> None:
        self.checks += 1
        print(f"  ok   {label}")

    def fail(self, label: str, detail: str) -> None:
        self.checks += 1
        self.failures.append(f"{label}: {detail}")
        print(f"  FAIL {label}: {detail}")

    def expect(self, label: str, actual, expected) -> None:
        if actual == expected:
            self.ok(label)
        else:
            self.fail(label, f"expected {expected!r}, observed {actual!r}")

    @property
    def passed(self) -> bool:
        return not self.failures


# --- Individual checks ------------------------------------------------------------------


def check_deployment_target(manifest: dict, f: Findings) -> None:
    platforms = manifest.get("platforms")
    if not isinstance(platforms, list):
        f.fail("deployment-target", f"`platforms` is not a list: {platforms!r}")
        return
    observed = [(p.get("platformName"), p.get("version")) for p in platforms]
    f.expect("deployment-target", observed, [("macos", MACOS_VERSION)])


def check_external_dependency(manifest: dict, f: Findings) -> None:
    deps = manifest.get("dependencies")
    if not isinstance(deps, list):
        f.fail("external-dependency", f"`dependencies` is not a list: {deps!r}")
        return
    if len(deps) != 1:
        f.fail(
            "external-dependency",
            f"expected exactly 1 direct dependency, observed {len(deps)}",
        )
        return

    entry = deps[0].get("sourceControl") if isinstance(deps[0], dict) else None
    if not isinstance(entry, list) or len(entry) != 1:
        f.fail("external-dependency", f"unrecognised dependency shape: {deps[0]!r}")
        return
    pin = entry[0]

    f.expect("external-dependency-identity", pin.get("identity"), DEPENDENCY_IDENTITY)

    location = pin.get("location") or {}
    remote = location.get("remote") or []
    url = remote[0].get("urlString") if remote else None
    f.expect("external-dependency-url", url, DEPENDENCY_URL)

    requirement = pin.get("requirement") or {}
    f.expect(
        "external-dependency-requirement",
        requirement,
        {"exact": [DEPENDENCY_EXACT]},
    )


def check_products(manifest: dict, f: Findings) -> None:
    products = manifest.get("products")
    if not isinstance(products, list):
        f.fail("products", f"`products` is not a list: {products!r}")
        return

    names = [p.get("name") for p in products]
    f.expect("product-set", sorted(names), sorted(REQUIRED_PRODUCTS))

    for product in products:
        name = product.get("name")
        if name not in REQUIRED_PRODUCTS:
            continue  # already reported by the set check
        expected_executable, expected_targets = REQUIRED_PRODUCTS[name]
        product_type = product.get("type")
        is_executable = isinstance(product_type, dict) and "executable" in product_type
        if is_executable == expected_executable:
            f.ok(f"product-type[{name}]")
        else:
            f.fail(
                f"product-type[{name}]",
                f"expected executable={expected_executable}, observed type {product_type!r}",
            )
        f.expect(f"product-targets[{name}]", product.get("targets"), expected_targets)


def check_targets(manifest: dict, f: Findings) -> None:
    targets = manifest.get("targets")
    if not isinstance(targets, list):
        f.fail("targets", f"`targets` is not a list: {targets!r}")
        return

    names = [t.get("name") for t in targets]
    f.expect("target-set", sorted(names), sorted(REQUIRED_TARGETS))

    for target in targets:
        name = target.get("name")
        if name not in REQUIRED_TARGETS:
            continue  # already reported by the set check
        expected_by_name, expected_products = REQUIRED_TARGETS[name]
        observed_by_name, observed_products = parse_edges(target, name, f)
        f.expect(
            f"target-edges[{name}]",
            sorted(observed_by_name),
            sorted(expected_by_name),
        )
        f.expect(
            f"target-product-edges[{name}]",
            sorted(observed_products),
            sorted(expected_products),
        )


def parse_edges(target: dict, name: str, f: Findings) -> tuple[list[str], list[str]]:
    by_name: list[str] = []
    products: list[str] = []
    for dep in target.get("dependencies") or []:
        if not isinstance(dep, dict) or len(dep) != 1:
            f.fail(f"target-edges[{name}]", f"unrecognised edge shape: {dep!r}")
            continue
        (kind, payload), = dep.items()
        if kind in ("byName", "target"):
            by_name.append(payload[0])
        elif kind == "product":
            products.append(payload[0])
        else:
            f.fail(f"target-edges[{name}]", f"unsupported edge kind {kind!r}")
    return by_name, products


def check_sources(manifest: dict, f: Findings) -> None:
    for target in manifest.get("targets") or []:
        name = target.get("name")
        if name not in REQUIRED_TARGETS:
            continue
        container = "Tests" if target.get("type") == "test" else "Sources"
        directory = PACKAGE_ROOT / container / name
        if not directory.is_dir():
            f.fail(f"sources[{name}]", f"{container}/{name}/ does not exist")
            continue
        sources = sorted(p for p in directory.rglob("*.swift") if p.is_file())
        if sources:
            f.ok(f"sources[{name}] ({len(sources)} file(s))")
        else:
            f.fail(f"sources[{name}]", f"no .swift file under {container}/{name}/")


# --- Entry point ------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <dump-package-json>", file=sys.stderr)
        return 2

    path = Path(argv[1])
    try:
        manifest = json.loads(path.read_text())
    except FileNotFoundError:
        print(f"check_package: no such file: {path}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"check_package: {path} is not valid JSON: {exc}", file=sys.stderr)
        return 2

    print(f"check_package: {path} (package root {PACKAGE_ROOT})")
    f = Findings()
    check_deployment_target(manifest, f)
    check_external_dependency(manifest, f)
    check_products(manifest, f)
    check_targets(manifest, f)
    check_sources(manifest, f)

    if f.passed:
        print(f"check_package: PASS ({f.checks} checks)")
        return 0

    print(f"check_package: FAIL ({len(f.failures)} of {f.checks} checks)", file=sys.stderr)
    for failure in f.failures:
        print(f"  - {failure}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
