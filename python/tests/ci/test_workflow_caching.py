# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Static checks for the GitHub Actions lane set, cache ownership, and runner guards."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = ROOT / ".github" / "workflows"
CONAN_CACHE_ACTION = ROOT / ".github" / "actions" / "conan-cache" / "action.yaml"
# Build switches and compile flags live in dedicated cmake/ modules (see CMakeLists.txt for the map).
CMAKE_OPTIONS_MODULE = ROOT / "cmake" / "AlayaOptions.cmake"
CMAKE_FLAGS_MODULE = ROOT / "cmake" / "AlayaFlags.cmake"
PYPROJECT = ROOT / "pyproject.toml"
PRINT_SUMMARY = ROOT / "cmake" / "PrintSummary.cmake"

# Filename == lane == display name is the whole contract; anything outside this set is drift.
LANE_FILES = {
    "build.yaml",
    "coverage.yaml",
    "lint.yaml",
    "release.yaml",
    "sanitizers.yaml",
    "tests.yaml",
    "wheels.yaml",
}
SHA_PINNED = re.compile(r"@[0-9a-f]{40}$")
# Trusted publishers may pin by major tag; everything else must pin a full commit SHA.
TAG_PIN_ALLOWED = ("actions/", "astral-sh/setup-uv@", "pypa/")


def _yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _triggers(workflow: dict) -> dict:
    # YAML 1.1 parses a bare `on:` key as boolean True.
    return workflow.get("on", workflow.get(True))


def _steps(workflow_name: str, job_name: str) -> list[dict]:
    workflow = _yaml(WORKFLOWS / workflow_name)
    return workflow["jobs"][job_name]["steps"]


def _uses(steps: list[dict], action: str) -> list[dict]:
    return [step for step in steps if step.get("uses") == action]


def _uses_matching(steps: list[dict], prefix: str) -> list[dict]:
    return [step for step in steps if str(step.get("uses", "")).startswith(prefix)]


def _named_step(steps: list[dict], name: str) -> dict:
    for step in steps:
        if step.get("name") == name:
            return step
    raise AssertionError(f"step not found: {name}")


def _all_workflow_steps() -> list[tuple[str, str, dict]]:
    return [
        (workflow_path.name, job_name, step)
        for workflow_path in sorted(WORKFLOWS.glob("*.yaml"))
        for workflow in [_yaml(workflow_path)]
        for job_name, job in workflow["jobs"].items()
        for step in job.get("steps", [])
    ]


def test_workflow_set_is_the_seven_lane_contract() -> None:
    assert {path.name for path in WORKFLOWS.glob("*.yaml")} == LANE_FILES


def test_no_workflow_reacts_to_pull_request_events() -> None:
    for name in LANE_FILES:
        triggers = _triggers(_yaml(WORKFLOWS / name))
        assert triggers is not None, name
        assert "pull_request" not in triggers, name
        assert "pull_request_target" not in triggers, name


def test_self_hosted_jobs_carry_the_repository_guard() -> None:
    for name in LANE_FILES:
        workflow = _yaml(WORKFLOWS / name)
        for job_name, job in workflow["jobs"].items():
            if "g05" in str(job.get("runs-on", "")):
                assert "huanglune/AlayaLite" in str(job.get("if", "")), f"{name}:{job_name}"


def test_third_party_actions_are_sha_pinned() -> None:
    for name, job_name, step in _all_workflow_steps():
        uses = str(step.get("uses", ""))
        if not uses or uses.startswith("./") or uses.startswith(TAG_PIN_ALLOWED):
            continue
        assert SHA_PINNED.search(uses), f"{name}:{job_name} uses unpinned {uses}"


def test_lint_job_does_not_use_conan_cache_action() -> None:
    # The lint lane never compiles C++, so it must not restore the Conan cache.
    assert not _uses(_steps("lint.yaml", "lint"), "./.github/actions/conan-cache")


def test_lint_has_dedicated_hook_environment_cache() -> None:
    cache_steps = _uses(_steps("lint.yaml", "lint"), "actions/cache@v5")

    assert any(step["with"]["path"] == "~/.cache/pre-commit" for step in cache_steps)
    assert any("hashFiles('.pre-commit-config.yaml')" in step["with"]["key"] for step in cache_steps)


def test_setup_uv_cache_keys_are_driven_by_uv_lock_only() -> None:
    setup_steps = [
        step for _, _, step in _all_workflow_steps() if str(step.get("uses", "")).startswith("astral-sh/setup-uv@")
    ]

    assert setup_steps
    assert all(step.get("with", {}).get("enable-cache") is True for step in setup_steps)
    assert all(step.get("with", {}).get("cache-dependency-glob") == "uv.lock" for step in setup_steps)


def test_conan_cache_action_uses_explicit_restore_and_save() -> None:
    action = _yaml(CONAN_CACHE_ACTION)
    uses_values = [step.get("uses") for step in action["runs"]["steps"]]

    assert "actions/cache/restore@v5" in uses_values
    assert "actions/cache/save@v5" in uses_values
    assert not any(uses and "@v4" in uses for uses in uses_values)


def test_conan_cache_key_is_defined_once_and_tracks_dependency_drivers() -> None:
    action = _yaml(CONAN_CACHE_ACTION)
    steps = action["runs"]["steps"]
    compute = _named_step(steps, "Compute Conan cache key")
    restore = next(step for step in steps if step.get("uses") == "actions/cache/restore@v5")
    save = next(step for step in steps if step.get("uses") == "actions/cache/save@v5")

    prefix = compute["env"]["CACHE_PREFIX"]
    dependency_hash = compute["env"]["DEPENDENCY_HASH"]
    assert prefix.startswith("conan-v3-")
    assert "${{ inputs.target-arch }}" in prefix
    # Dependency resolution goes through the vendored Conan provider; the cache key must track it.
    assert "conanfile.py" in dependency_hash
    assert "cmake/vendor/conan_provider.cmake" in dependency_hash
    # Restore and save must consume the same computed key: one definition, no drift.
    assert restore["with"]["key"] == save["with"]["key"] == "${{ steps.cache-key.outputs.key }}"


def test_workflow_conan_cache_calls_are_arch_scoped_and_nonfatal_on_save_race() -> None:
    conan_steps = [step for _, _, step in _all_workflow_steps() if step.get("uses") == "./.github/actions/conan-cache"]
    stale_steps = [
        step for _, _, step in _all_workflow_steps() if step.get("uses") == "./.github/actions/cache-restore"
    ]

    assert conan_steps
    assert not stale_steps
    assert all("target-arch" in step.get("with", {}) for step in conan_steps)
    assert all(
        step.get("continue-on-error") is True for step in conan_steps if step.get("with", {}).get("mode") == "save"
    )


def test_self_hosted_lanes_do_not_use_hosted_cache_actions() -> None:
    for name in ("tests.yaml", "build.yaml", "sanitizers.yaml"):
        workflow = _yaml(WORKFLOWS / name)
        for job_name, job in workflow["jobs"].items():
            for step in job["steps"]:
                uses = str(step.get("uses", ""))
                assert not uses.startswith("actions/cache"), f"{name}:{job_name}"
                assert "ccache-action" not in uses, f"{name}:{job_name}"
                assert uses != "./.github/actions/conan-cache", f"{name}:{job_name}"


def test_coverage_jobs_enable_ccache() -> None:
    python_steps = _steps("coverage.yaml", "codecov-python")
    cpp_steps = _steps("coverage.yaml", "codecov-cpp")

    assert _uses_matching(python_steps, "hendrikmuhs/ccache-action@")
    assert _uses_matching(cpp_steps, "hendrikmuhs/ccache-action@")
    assert "CMAKE_CXX_COMPILER_LAUNCHER=ccache" in _named_step(python_steps, "Build Python coverage environment")["run"]
    assert "CMAKE_CXX_COMPILER_LAUNCHER=ccache" in _named_step(cpp_steps, "Run c++ code coverage")["run"]


def test_coverage_python_gate_uploads_without_flakes() -> None:
    coverage_steps = _steps("coverage.yaml", "codecov-python")
    run_step = _named_step(coverage_steps, "Run python code coverage")
    upload_step = _named_step(coverage_steps, "Upload Python coverage to Codecov")

    assert "python_coverage_with_crash_diagnostics.sh" in run_step["run"]
    assert upload_step["continue-on-error"] is True
    assert str(upload_step["uses"]).startswith("codecov/codecov-action@")


def test_codecov_python_logs_laser_simd_selection() -> None:
    codecov_script = (ROOT / ".github" / "scripts" / "codecov" / "python_coverage_with_crash_diagnostics.sh").read_text(
        encoding="utf-8"
    )

    assert "laser_simd=" in codecov_script
    assert "laser.selected_simd()" in codecov_script


def test_codecov_cpp_covers_laser_simd_dispatch_via_labels() -> None:
    # The C++ coverage build derives its targets from CTest labels rather than a hard-coded
    # list, so "is the LASER SIMD dispatch test covered?" is answered by its label being in
    # the coverage label set -- not by a literal target name in the script.
    codecov_script = (ROOT / ".github" / "scripts" / "codecov" / "gnu_codecoverage.sh").read_text(encoding="utf-8")
    assert "--show-only=json-v1" in codecov_script
    assert '-L "${CTEST_LABELS}"' in codecov_script

    cpp_env = _yaml(WORKFLOWS / "coverage.yaml")["jobs"]["codecov-cpp"]["env"]
    # `ctest -L` treats the expression as a regex matched against each label of a
    # test, selecting the test if any label matches -- model that, not a literal
    # set intersection (the current expression "." matches every labeled test).
    coverage_expr = re.compile(str(cpp_env["CTEST_LABELS"]))

    laser_cmake = (ROOT / "tests" / "laser" / "CMakeLists.txt").read_text(encoding="utf-8")
    registration = re.search(r"alaya_add_test\([^)]*laser_simd_dispatch_test[^)]*\)", laser_cmake)
    assert registration, "laser_simd_dispatch_test is not registered as a CTest"
    labels = re.search(r"LABELS\s+([A-Za-z0-9_ ]+)", registration.group(0))
    assert labels, "laser_simd_dispatch_test has no LABELS"
    assert any(coverage_expr.search(label) for label in labels.group(1).split()), (
        "laser_simd_dispatch_test is not selected by the C++ coverage CTest labels"
    )


def test_codecov_python_build_uses_unit_build_configuration() -> None:
    coverage_build = _named_step(_steps("coverage.yaml", "codecov-python"), "Build Python coverage environment")["run"]

    assert "CMAKE_CXX_COMPILER_LAUNCHER=ccache" in coverage_build
    assert "-DALAYA_NATIVE_ARCH=OFF" in coverage_build
    assert "uv sync --dev --locked" in coverage_build
    assert "CXXFLAGS" not in coverage_build
    assert "install.strip" not in coverage_build


def test_coverage_runs_nightly_and_on_demand_only() -> None:
    triggers = _triggers(_yaml(WORKFLOWS / "coverage.yaml"))

    assert triggers is not None
    assert set(triggers) == {"workflow_dispatch", "schedule"}
    assert triggers["schedule"] == [{"cron": "0 19 * * *"}]


def test_ccache_builds_disable_native_arch() -> None:
    """Do not cache -march=native objects across heterogeneous GitHub runners."""

    coverage_steps = _steps("coverage.yaml", "codecov-python")
    wheel_env = _yaml(WORKFLOWS / "wheels.yaml")["jobs"]["build_wheels"]["env"]
    codecov_script = (ROOT / ".github" / "scripts" / "codecov" / "gnu_codecoverage.sh").read_text(encoding="utf-8")

    assert "-DALAYA_NATIVE_ARCH=OFF" in _named_step(coverage_steps, "Build Python coverage environment")["run"]
    assert "-DALAYA_NATIVE_ARCH=OFF" in wheel_env["CMAKE_ARGS"]
    assert "-DALAYA_NATIVE_ARCH=OFF" in codecov_script


def test_cibuildwheel_builds_portable_package_targets() -> None:
    wheel_env = _yaml(WORKFLOWS / "wheels.yaml")["jobs"]["build_wheels"]["env"]
    cmake_args = wheel_env["CMAKE_ARGS"]

    assert "-DALAYA_NATIVE_ARCH=OFF" in cmake_args
    assert "-DBUILD_TOOLS=OFF" in cmake_args


def test_cibuildwheel_smoke_checks_the_1_1_public_surface() -> None:
    pyproject_text = PYPROJECT.read_text(encoding="utf-8")
    smoke_text = (ROOT / "python" / "tests" / "wheel" / "test_qg_platform_contract.py").read_text(encoding="utf-8")

    assert "python/tests/wheel/test_qg_platform_contract.py" in pyproject_text
    assert '"Index" not in alayalite.__all__' in smoke_text
    assert '"DiskCollection" not in alayalite.__all__' in smoke_text
    assert 'alayalite.__version__ == "1.1.0"' in smoke_text
    assert "Flat fallback is disabled" in smoke_text


def test_ccache_keys_are_versioned_for_portable_isa_reset() -> None:
    ccache_steps = [
        _uses_matching(_steps("coverage.yaml", "codecov-python"), "hendrikmuhs/ccache-action@")[0],
        _uses_matching(_steps("coverage.yaml", "codecov-cpp"), "hendrikmuhs/ccache-action@")[0],
    ]

    assert "portable-v3" in ccache_steps[0]["with"]["key"]
    assert "portable-v2" in ccache_steps[1]["with"]["key"]
    assert all("portable-v" in step["with"]["restore-keys"] for step in ccache_steps)


def test_codecov_python_restores_legacy_unit_test_cache() -> None:
    ccache_step = _uses_matching(_steps("coverage.yaml", "codecov-python"), "hendrikmuhs/ccache-action@")[0]

    assert "codecov-python-portable-v3" in ccache_step["with"]["key"]
    assert "py-unit-portable-v2" in ccache_step["with"]["restore-keys"]


def test_cmake_defaults_to_portable_native_arch_and_guards_packages() -> None:
    cmake_text = CMAKE_OPTIONS_MODULE.read_text(encoding="utf-8")

    assert 'option(ALAYA_NATIVE_ARCH "Compile with -march=native for host-specific builds" OFF)' in cmake_text
    assert "option(ALAYA_ALLOW_NATIVE_PACKAGE" in cmake_text
    assert "if(BUILD_PYTHON" in cmake_text
    assert "AND ALAYA_NATIVE_ARCH" in cmake_text
    assert "AND NOT ALAYA_ALLOW_NATIVE_PACKAGE" in cmake_text


def test_native_arch_option_name_is_used_consistently() -> None:
    pyproject_text = PYPROJECT.read_text(encoding="utf-8")
    summary_text = PRINT_SUMMARY.read_text(encoding="utf-8")

    assert 'ALAYA_NATIVE_ARCH="OFF"' in pyproject_text
    assert "${ALAYA_NATIVE_ARCH}" in summary_text
    assert "ALAYA_ENABLE_NATIVE_OPT" not in pyproject_text
    assert "ALAYA_ENABLE_NATIVE_OPT" not in summary_text


def test_portable_cmake_uses_avx2_baseline_without_native_or_avx512() -> None:
    cmake_text = CMAKE_FLAGS_MODULE.read_text(encoding="utf-8")

    assert "list(APPEND ALAYA_SIMD_COMPILE_OPTIONS -mavx2 -mfma)" in cmake_text
    assert "list(APPEND ALAYA_SIMD_COMPILE_OPTIONS -mavx512" not in cmake_text
