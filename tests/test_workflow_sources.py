"""A source snapshot must include the local modules its workflows import."""

import ast
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def _module_path(parts):
    path = ROOT.joinpath(*parts)
    return path.with_suffix(".py") if path.with_suffix(".py").is_file() else path


def _package_names(path):
    """Names exported by a package may be attributes rather than submodules."""
    initializer = path / "__init__.py"
    if not initializer.is_file():
        return set()
    names = set()
    for node in ast.walk(ast.parse(initializer.read_text())):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
    return names


def _source_trees():
    files = [*ROOT.glob("*.py"), *(ROOT / "Utils").rglob("*.py"),
             *(ROOT / "scripts").rglob("*.py")]
    for path in sorted(files):
        yield path, ast.parse(path.read_text(), filename=str(path))
    for path in sorted(ROOT.glob("*.ipynb")):
        for cell in json.loads(path.read_text())["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            if source.lstrip().startswith("%%"):
                continue
            source = "".join(line for line in source.splitlines(keepends=True)
                             if not line.lstrip().startswith(("%", "!")))
            yield path, ast.parse(source, filename=str(path))


def test_project_imports_resolve_in_the_source_snapshot():
    local_roots = {"Utils", "scripts", "preprocessing", "si_readers"}
    local_roots.update(path.stem for path in ROOT.glob("*.py"))
    missing = []
    for source, tree in _source_trees():
        package = list(source.relative_to(ROOT).parts[:-1])
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name.split(".") for alias in node.names]
                for parts in modules:
                    if parts[0] in local_roots and not _module_path(parts).exists():
                        missing.append(f"{source.relative_to(ROOT)}:{node.lineno}: {'.'.join(parts)}")
            elif isinstance(node, ast.ImportFrom):
                parts = node.module.split(".") if node.module else []
                if node.level:
                    parts = package[:len(package) - node.level + 1] + parts
                if not parts or (not node.level and parts[0] not in local_roots):
                    continue
                target = _module_path(parts)
                if not target.exists():
                    missing.append(f"{source.relative_to(ROOT)}:{node.lineno}: {'.'.join(parts)}")
                elif target.is_dir():
                    exports = _package_names(target)
                    for alias in node.names:
                        if alias.name != "*" and alias.name not in exports:
                            child = parts + [alias.name]
                            if not _module_path(child).exists():
                                missing.append(f"{source.relative_to(ROOT)}:{node.lineno}: {'.'.join(child)}")
    assert not missing, "Missing local workflow modules:\n" + "\n".join(missing)


def test_pipeline_entry_sources_are_included():
    entries = (
        "scripts/run_pipeline.sh", "scripts/generate_pipeline_config.py",
        "scripts/run_tuning_curves.py", "scripts/run_cylinder_spatial.py",
        "scripts/plot_hd_cell_distribution.py", "preprocessing/spikeinterface/app.py",
        "preprocessing/spikeinterface/canonical_unit_artifacts.py",
        "preprocessing/spikeinterface/make_analyzer_for_sigui.py",
        "preprocessing/spikeinterface/split_kilosort_by_recording.py",
        "preprocessing/spikeinterface/generate_adc_spike_time.py",
        "ebc_video_rectangle.py", "ebc_video_circle.py", "locate_rf.py",
    )
    missing = [entry for entry in entries if not (ROOT / entry).is_file()]
    assert not missing, f"Missing pipeline entries: {missing}"


def test_pipeline_shell_uses_included_python_entry_modules():
    runner = ROOT / "scripts/run_pipeline.sh"
    subprocess.run(["bash", "-n", str(runner)], check=True, capture_output=True, text=True)
    modules = set(re.findall(r"\s-m\s+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)", runner.read_text()))
    required = {
        "scripts.generate_pipeline_config", "scripts.run_tuning_curves",
        "scripts.plot_hd_cell_distribution", "scripts.run_cylinder_spatial",
        "scripts.spatial_cell_analysis", "scripts.ebc_video",
        "preprocessing.spikeinterface.app", "preprocessing.spikeinterface.generate_adc_spike_time",
    }
    assert required <= modules, f"Pipeline does not invoke the expected entry modules: {required - modules}"
    for module in modules:
        if module.startswith(("scripts.", "preprocessing.")):
            assert _module_path(module.split(".")).exists(), f"Missing pipeline command module: {module}"
