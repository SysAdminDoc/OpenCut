"""Every install into ``<data folder>/packages`` goes through one store.

The store keeps the copy of a shared dependency it already has when the new
packages accept it, refuses an install that would break a package that stays,
merges shared folders (``nvidia``, ``__pycache__``) entry by entry, puts back
a merge that was killed partway, and moves out packages built for another
Python once per runtime.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

from opencut import helpers, security
from opencut.core import package_store
from opencut.dependency_support import DECLARED_REQUIREMENTS
from packaging import tags

REPO_ROOT = Path(__file__).resolve().parents[1]
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "packages").mkdir(parents=True)
    monkeypatch.setattr(helpers, "OPENCUT_DIR", str(data))
    monkeypatch.setattr(security, "install_python", lambda: "C:/Python313/python.exe")
    return data


def _dist(
    folder, name, version, files=None, *, requires=(), wheel_tags=("py3-none-any",), top_level=(), requested=False
):
    """Write what ``pip install --target`` leaves for one wheel, RECORD included."""
    folder = Path(folder)
    info = folder / f"{re.sub(r'[-.]+', '_', name)}-{version}.dist-info"
    info.mkdir(parents=True)
    if requested:  # pip marks the projects it was asked for
        (info / "REQUESTED").write_bytes(b"")
    meta = ["Metadata-Version: 2.1", f"Name: {name}", f"Version: {version}"]
    meta += [f"Requires-Dist: {req}" for req in requires]
    (info / "METADATA").write_text("\n".join(meta) + "\n", encoding="utf-8")
    if wheel_tags is not None:
        lines = ["Wheel-Version: 1.0", *(f"Tag: {tag}" for tag in wheel_tags)]
        (info / "WHEEL").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if top_level:
        (info / "top_level.txt").write_text("\n".join(top_level) + "\n", encoding="utf-8")
    record = []
    for rel, content in (files or {}).items():
        path = folder / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        record.append("../../bin/" + rel[4:] if rel.startswith("bin/") else rel)
    record += [f"{info.name}/{entry.name}" for entry in info.iterdir()] + [f"{info.name}/RECORD"]
    (info / "RECORD").write_text("".join(f"{rel},,\n" for rel in record), encoding="utf-8")
    return info


def _tree(root):
    return {
        p.relative_to(root).as_posix(): p.read_bytes() if p.is_file() else None
        for p in sorted(Path(root).rglob("*"))
    }


class _Pip:
    """pip fills --target through ``stage(folder)``; the child server's import is scripted."""

    def __init__(self, stage, import_exit=0):
        self.stage = stage
        self.import_exit = import_exit
        self.commands = []

    def __call__(self, cmd, timeout, on_start=None):
        self.commands.append(cmd)
        if "pip" in cmd:
            self.stage(Path(cmd[cmd.index("--target") + 1]))
            return subprocess.CompletedProcess(cmd, 0, "", "")
        assert "--import-check" in cmd
        return subprocess.CompletedProcess(cmd, self.import_exit, "", "Fatal Python error")

    @property
    def import_checks(self):
        return [cmd[cmd.index("--import-check") + 1] for cmd in self.commands if "--import-check" in cmd]


def _pip(monkeypatch, stage, **kwargs):
    fake = _Pip(stage, **kwargs)
    monkeypatch.setattr(package_store, "_run", fake)
    return fake


def _leftovers(data_dir):
    return sorted(
        p.name for p in data_dir.iterdir() if p.name.startswith(".")
    )


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------
def test_declared_requirements_are_the_pyproject_extras():
    extras = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["optional-dependencies"]
    declared = {}
    for group in extras.values():
        for line in group:
            body = line.split(";", 1)[0].strip()
            name = package_store.canonical(Requirement(body).name)
            if name != "opencut-ppro":
                declared.setdefault(name, set()).add(body)
    assert {name: bodies for name, bodies in declared.items() if len(bodies) > 1} == {}
    assert dict(DECLARED_REQUIREMENTS) == {name: bodies.pop() for name, bodies in declared.items()}


@pytest.mark.parametrize(
    "asked, installed",
    [
        ("mediapipe", "mediapipe>=0.10,<2"),
        ("faster_whisper", "faster-whisper>=1.1,<2"),
        ("pyannote.audio", "pyannote.audio>=4.0,<5"),
        ("transformers", "transformers>=5.10"),
        ("mediapipe==0.10.14", "mediapipe==0.10.14"),  # an explicit pin is the caller's
        ("rembg[gpu]", "rembg[gpu]>=2.0.82,<3"),
        ("not-in-pyproject", "not-in-pyproject"),
    ],
)
def test_a_bare_name_gets_the_bounds_pyproject_declares(asked, installed):
    assert security.runtime_security_requirement(asked) == installed


def test_the_face_install_route_asks_pip_for_a_bounded_mediapipe(monkeypatch, data_dir):
    """/video/face/install passes the bare name "mediapipe" to safe_pip_install."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    fake = _pip(monkeypatch, lambda target: _dist(target, "mediapipe", "1.0.0", {"mediapipe/__init__.py": b""}))

    security.safe_pip_install("mediapipe")

    [pip] = [cmd for cmd in fake.commands if "pip" in cmd]
    assert "mediapipe>=0.10,<2" in pip
    assert "mediapipe" not in pip
    assert fake.import_checks == ["mediapipe"]


# ---------------------------------------------------------------------------
# Shared dependencies
# ---------------------------------------------------------------------------
def test_a_dependency_the_store_has_stays_when_the_new_package_accepts_it(monkeypatch, data_dir):
    packages = data_dir / "packages"
    _dist(packages, "old-tool", "1.0", {"old_tool/__init__.py": b"old"}, requires=["shared>=1,<2"])
    _dist(packages, "shared", "1.0", {"shared/__init__.py": b"shared 1.0"})

    def stage(target):
        _dist(target, "new-tool", "1.0", {"new_tool/__init__.py": b"new"}, requires=["shared>=1"])
        _dist(target, "shared", "1.5", {"shared/__init__.py": b"shared 1.5"})

    fake = _pip(monkeypatch, stage)

    package_store.install(["new-tool"])

    assert (packages / "shared" / "__init__.py").read_bytes() == b"shared 1.0"
    assert (packages / "shared-1.0.dist-info").is_dir()
    assert not (packages / "shared-1.5.dist-info").exists()
    assert (packages / "new_tool" / "__init__.py").read_bytes() == b"new"
    assert fake.import_checks == ["new_tool"]
    assert _leftovers(data_dir) == []


def test_a_dependency_the_new_package_needs_newer_is_replaced_when_nothing_objects(monkeypatch, data_dir):
    packages = data_dir / "packages"
    _dist(packages, "shared", "1.0", {"shared/__init__.py": b"shared 1.0", "shared/gone.py": b""})

    def stage(target):
        _dist(target, "new-tool", "1.0", {"new_tool/__init__.py": b"new"}, requires=["shared>=1.5"])
        _dist(target, "shared", "1.5", {"shared/__init__.py": b"shared 1.5"})

    _pip(monkeypatch, stage)

    package_store.install(["new-tool"])

    assert (packages / "shared" / "__init__.py").read_bytes() == b"shared 1.5"
    assert not (packages / "shared" / "gone.py").exists()
    assert sorted(p.name for p in packages.glob("shared-*.dist-info")) == ["shared-1.5.dist-info"]


def test_a_replacement_that_would_break_a_package_that_stays_is_refused(monkeypatch, data_dir):
    packages = data_dir / "packages"
    _dist(packages, "old-tool", "1.0", {"old_tool/__init__.py": b"old"}, requires=["shared<2"])
    _dist(packages, "shared", "1.0", {"shared/__init__.py": b"shared 1.0"})
    before = _tree(packages)

    def stage(target):
        _dist(target, "new-tool", "1.0", {"new_tool/__init__.py": b"new"}, requires=["shared>=2"])
        _dist(target, "shared", "2.0", {"shared/__init__.py": b"shared 2.0"})

    fake = _pip(monkeypatch, stage)

    with pytest.raises(RuntimeError) as caught:
        package_store.install(["new-tool"])

    assert str(caught.value).startswith("Nothing was changed, because installing new-tool")
    assert "old-tool 1.0 needs shared<2, and this would replace shared 1.0 with 2.0" in str(caught.value)
    assert _tree(packages) == before
    assert fake.import_checks == []
    assert _leftovers(data_dir) == []


def test_a_reinstall_replaces_shared_dependencies_too(monkeypatch, data_dir):
    packages = data_dir / "packages"
    _dist(packages, "shared", "1.0", {"shared/__init__.py": b"shared 1.0"})

    def stage(target):
        _dist(target, "new-tool", "1.0", {"new_tool/__init__.py": b"new"}, requires=["shared>=1"])
        _dist(target, "shared", "1.5", {"shared/__init__.py": b"shared 1.5"})

    _pip(monkeypatch, stage)

    package_store.install(["new-tool"], keep_shared=False)

    assert (packages / "shared" / "__init__.py").read_bytes() == b"shared 1.5"


def test_a_downgrade_takes_the_newer_dist_info_out(monkeypatch, data_dir):
    packages = data_dir / "packages"
    _dist(packages, "alpha", "2.0", {"alpha/__init__.py": b"2.0"})
    _pip(monkeypatch, lambda target: _dist(target, "alpha", "1.0", {"alpha/__init__.py": b"1.0"}))

    package_store.install(["alpha==1.0"])

    assert sorted(p.name for p in packages.iterdir()) == ["alpha", "alpha-1.0.dist-info"]
    assert (packages / "alpha" / "__init__.py").read_bytes() == b"1.0"


# ---------------------------------------------------------------------------
# Shared folders
# ---------------------------------------------------------------------------
def test_shared_folders_keep_the_files_of_packages_that_stay(monkeypatch, data_dir):
    """pip --target --upgrade swaps whole top-level folders, so nvidia/cublas went with nvidia/."""
    packages = data_dir / "packages"
    pyc = f"__pycache__/typing_extensions.cpython-{sys.version_info[0]}{sys.version_info[1]}.pyc"
    _dist(packages, "nvidia-cublas", "12.0", {"nvidia/cublas/lib/cublas.dll": b"cublas"})
    _dist(packages, "typing-extensions", "4.0", {"typing_extensions.py": b"te", pyc: b"te pyc"})
    _dist(packages, "other", "1.0", {"bin/other.exe": b"other"})

    def stage(target):
        _dist(target, "nvidia-cudnn", "9.0", {
            "nvidia/cudnn/lib/cudnn.dll": b"cudnn",
            "__pycache__/six.cpython-313.pyc": b"six pyc",
            "bin/cudnn.exe": b"cudnn exe",
        })

    _pip(monkeypatch, stage)

    package_store.install(["nvidia-cudnn"], modules=["nvidia.cudnn"])

    assert (packages / "nvidia" / "cublas" / "lib" / "cublas.dll").read_bytes() == b"cublas"
    assert (packages / "nvidia" / "cudnn" / "lib" / "cudnn.dll").read_bytes() == b"cudnn"
    assert (packages / pyc).read_bytes() == b"te pyc"
    assert (packages / "__pycache__" / "six.cpython-313.pyc").exists()
    assert sorted(p.name for p in (packages / "bin").iterdir()) == ["cudnn.exe", "other.exe"]


def test_the_import_check_names_the_module_a_project_installs():
    def named(name, files, top_level=()):
        return package_store.Dist(
            name=name, display=name, version="1", info=f"{name}-1.dist-info", requires=(),
            files=frozenset(files), top_level=tuple(top_level), tags=(),
        )

    assert package_store.import_names(
        named("pyannote-audio", ["pyannote/audio/__init__.py", "pyannote/audio/core/__init__.py"])
    ) == ["pyannote.audio"]
    assert package_store.import_names(
        named("opencv-python", ["cv2/__init__.py", "cv2/data/__init__.py"], ["cv2"])
    ) == ["cv2"]
    assert package_store.import_names(named("pillow", ["PIL/__init__.py", "PIL/Image.py"])) == ["PIL"]
    assert package_store.import_names(named("pyyaml", ["yaml/__init__.py", "_yaml/__init__.py"])) == ["yaml"]
    assert package_store.import_names(named("six", ["six.py"])) == ["six"]


# ---------------------------------------------------------------------------
# A killed merge
# ---------------------------------------------------------------------------
_KILL_PARTWAY = """\
import os, sys
from opencut.core import package_store

real_replace = os.replace
calls = []

def replace(src, dst):
    calls.append(dst)
    if len(calls) == 4:  # the new alpha is in, the old dist-info is aside, then the power goes
        os._exit(9)
    real_replace(src, dst)

os.replace = replace
package_store.merge_staged_packages(sys.argv[1], sys.argv[2])
"""


def test_a_merge_killed_partway_is_put_back_by_the_next_install(monkeypatch, data_dir, tmp_path):
    packages = data_dir / "packages"
    _dist(packages, "alpha", "1.0", {"alpha/__init__.py": b"old alpha", "bin/alpha.exe": b"old exe"})
    _dist(packages, "beta", "1.0", {"bin/beta.exe": b"beta"})
    before = _tree(packages)
    staging = data_dir / ".alpha-staging-k1ll"
    _dist(staging, "alpha", "2.0", {"alpha/__init__.py": b"new alpha", "bin/alpha.exe": b"new exe"})
    script = tmp_path / "kill_partway.py"
    script.write_text(_KILL_PARTWAY, encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")])))

    killed = subprocess.run(
        [sys.executable, str(script), str(staging), str(packages)],
        env=env, capture_output=True, text=True, timeout=120, creationflags=NO_WINDOW,
    )

    assert killed.returncode == 9, killed.stderr
    assert (packages / "alpha" / "__init__.py").read_bytes() == b"new alpha"
    assert not (packages / "alpha-1.0.dist-info").exists()  # broken until someone puts it back
    [aside] = [p for p in data_dir.iterdir() if p.name.startswith(".packages-replaced-")]
    assert (aside / "journal").is_file()

    _pip(monkeypatch, lambda target: _dist(target, "gamma", "1.0", {"gamma/__init__.py": b"gamma"}))
    package_store.install(["gamma"])

    after = _tree(packages)
    assert {rel: data for rel, data in after.items() if not rel.startswith("gamma")} == before
    assert (packages / "gamma" / "__init__.py").exists()
    assert not [p for p in data_dir.iterdir() if p.name.startswith(".packages-replaced-")]


# ---------------------------------------------------------------------------
# Packages built for another Python
# ---------------------------------------------------------------------------
def _other_python_tag():
    here = next(iter(tags.sys_tags()))
    minor = sys.version_info[1] - 1
    return f"cp3{minor}-cp3{minor}-{here.platform}"


def test_packages_built_for_another_python_are_moved_out_once(data_dir):
    packages = data_dir / "packages"
    best = next(iter(tags.sys_tags()))
    _dist(packages, "native", "1.0", {"native/__init__.py": b"n", "bin/native.exe": b"n"},
          wheel_tags=[_other_python_tag()])
    _dist(packages, "fits", "1.0", {"fits/__init__.py": b"f"}, wheel_tags=[str(best)])
    _dist(packages, "pure", "1.0", {"pure/__init__.py": b"p", "bin/pure.exe": b"p"})
    _dist(packages, "untagged", "1.0", {"untagged/__init__.py": b"u"}, wheel_tags=None)
    lines = []

    with package_store.store_lock():
        moved = package_store.inventory_once(lines.append)

    assert moved == ["native"]
    assert sorted(p.name for p in packages.iterdir()) == [
        "bin", "fits", "fits-1.0.dist-info", "pure", "pure-1.0.dist-info",
        "untagged", "untagged-1.0.dist-info",
    ]
    assert sorted(p.name for p in (packages / "bin").iterdir()) == ["pure.exe"]
    [quarantine] = (data_dir / "packages-quarantine").iterdir()
    assert (quarantine / "native" / "__init__.py").read_bytes() == b"n"
    assert (quarantine / "bin" / "native.exe").exists()
    readme = (quarantine / "README.txt").read_text(encoding="utf-8")
    assert f"native 1.0 ({_other_python_tag()})" in readme
    assert security.runtime_abi() in readme
    marker = json.loads((data_dir / "packages-inventory.json").read_text(encoding="utf-8"))
    assert marker["abi"] == security.runtime_abi()
    assert marker["moved"] == ["native"]
    assert marker["kept"] == ["fits", "pure", "untagged"]
    assert lines and str(quarantine) in lines[-1]

    _dist(packages, "native", "1.0", {"native/__init__.py": b"n"}, wheel_tags=[_other_python_tag()])
    with package_store.store_lock():
        assert package_store.inventory_once(lines.append) == []  # once per runtime


# ---------------------------------------------------------------------------
# One install at a time
# ---------------------------------------------------------------------------
_HOLD_LOCK = """\
import sys
from filelock import FileLock

lock = FileLock(sys.argv[1])
lock.acquire()
print("held", flush=True)
sys.stdin.read()
lock.release()
"""


@pytest.fixture
def another_install(data_dir, tmp_path):
    script = tmp_path / "hold_lock.py"
    script.write_text(_HOLD_LOCK, encoding="utf-8")
    holder = subprocess.Popen(
        [sys.executable, str(script), str(data_dir / "packages.lock")],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, creationflags=NO_WINDOW,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        yield
    finally:
        holder.stdin.close()
        holder.wait(timeout=30)


def test_an_install_waits_for_another_one_then_says_so(another_install, monkeypatch, data_dir):
    lines = []
    fake = _pip(monkeypatch, lambda target: pytest.fail("ran pip while another install held the store"))
    # install() waits LOCK_WAIT (30 minutes) by default; the test waits a moment.
    monkeypatch.setattr(package_store.store_lock.__wrapped__, "__defaults__", (package_store._log, 0.3))

    with pytest.raises(package_store.StoreBusy, match="Another add-on install was still running"):
        package_store.install(["new-tool"], report=lines.append)

    assert lines == ["Waiting for another add-on install to finish..."]
    assert fake.commands == []
    assert _leftovers(data_dir) == []


def test_the_startup_check_skips_when_an_install_holds_the_store(another_install, data_dir):
    packages = data_dir / "packages"
    _dist(packages, "native", "1.0", {"native/__init__.py": b"n"}, wheel_tags=[_other_python_tag()])

    package_store.inventory_at_startup()

    assert (packages / "native" / "__init__.py").exists()
    assert not (data_dir / "packages-inventory.json").exists()


# ---------------------------------------------------------------------------
# Removing an add-on
# ---------------------------------------------------------------------------
def _two_add_ons(packages):
    """edge-tts and mediapipe as an install leaves them, sharing certifi and bin/."""
    _dist(packages, "edge-tts", "7.0", {"edge_tts/__init__.py": b"tts", "bin/edge-tts.exe": b"tts exe"},
          requires=["aiohttp>=3", "certifi"], requested=True)
    _dist(packages, "aiohttp", "3.9", {"aiohttp/__init__.py": b"aiohttp"}, requires=["yarl"])
    _dist(packages, "yarl", "1.9", {"yarl/__init__.py": b"yarl"})
    _dist(packages, "certifi", "2026.1", {"certifi/__init__.py": b"certifi"})
    _dist(packages, "mediapipe", "0.10", {"mediapipe/__init__.py": b"mp", "bin/mp.exe": b"mp exe"},
          requires=["numpy", "certifi"], requested=True)
    _dist(packages, "numpy", "2.1", {"numpy/__init__.py": b"numpy"})


def test_removing_an_add_on_takes_the_dependencies_only_it_needed(data_dir):
    packages = data_dir / "packages"
    _two_add_ons(packages)
    lines = []

    removed = package_store.remove(["Edge_TTS"], report=lines.append)

    assert removed == ["aiohttp 3.9", "edge-tts 7.0", "yarl 1.9"]
    assert sorted(package_store.read_dists(str(packages))) == ["certifi", "mediapipe", "numpy"]
    assert not (packages / "edge_tts").exists() and not (packages / "aiohttp").exists()
    assert sorted(p.name for p in (packages / "bin").iterdir()) == ["mp.exe"]
    assert (packages / "certifi" / "__init__.py").read_bytes() == b"certifi"
    assert lines == [f"Removed aiohttp 3.9, edge-tts 7.0, yarl 1.9 from {packages}"]
    assert _leftovers(data_dir) == []


def test_a_project_installed_for_itself_stays_when_the_add_on_that_needed_it_goes(data_dir):
    packages = data_dir / "packages"
    _dist(packages, "tool", "1.0", {"tool/__init__.py": b"t"}, requires=["numpy"], requested=True)
    _dist(packages, "numpy", "2.1", {"numpy/__init__.py": b"numpy"}, requested=True)

    assert package_store.remove(["tool"]) == ["tool 1.0"]
    assert sorted(package_store.read_dists(str(packages))) == ["numpy"]


def test_a_requirement_behind_an_extra_keeps_its_library(data_dir):
    """Nothing records which extras an install asked for, so an optional need counts."""
    packages = data_dir / "packages"
    _dist(packages, "tool", "1.0", {"tool/__init__.py": b"t"}, requires=["numpy"], requested=True)
    _dist(packages, "other", "1.0", {"other/__init__.py": b"o"}, requires=['numpy; extra == "fast"'],
          requested=True)
    _dist(packages, "numpy", "2.1", {"numpy/__init__.py": b"numpy"})

    assert package_store.remove(["tool"]) == ["tool 1.0"]
    assert (packages / "numpy" / "__init__.py").exists()


def test_an_add_on_that_another_one_needs_is_not_removed(data_dir):
    packages = data_dir / "packages"
    _two_add_ons(packages)
    before = _tree(packages)

    with pytest.raises(RuntimeError, match="Nothing was removed, because mediapipe still needs numpy"):
        package_store.remove(["numpy", "edge-tts"])

    assert _tree(packages) == before
    assert _leftovers(data_dir) == []


def test_removal_keeps_the_files_shared_folders_hold_for_packages_that_stay(data_dir):
    packages = data_dir / "packages"
    _dist(packages, "nvidia-cublas", "12.0", {"nvidia/cublas/lib/cublas.dll": b"cublas"}, requested=True)
    _dist(packages, "nvidia-cudnn", "9.0", {
        "nvidia/cudnn/lib/cudnn.dll": b"cudnn",
        "__pycache__/six.cpython-313.pyc": b"six pyc",
    }, requested=True)
    _dist(packages, "typing-extensions", "4.0", {
        "typing_extensions.py": b"te", "__pycache__/typing_extensions.cpython-313.pyc": b"te pyc",
    }, requested=True)

    assert package_store.remove(["nvidia-cudnn"]) == ["nvidia-cudnn 9.0"]

    assert (packages / "nvidia" / "cublas" / "lib" / "cublas.dll").read_bytes() == b"cublas"
    assert not (packages / "nvidia" / "cudnn").exists()
    assert sorted(p.name for p in (packages / "__pycache__").iterdir()) == ["typing_extensions.cpython-313.pyc"]


def test_a_name_the_store_does_not_have_changes_nothing(data_dir):
    packages = data_dir / "packages"
    _two_add_ons(packages)
    before = _tree(packages)

    assert package_store.remove(["whisperx"]) == []
    assert _tree(packages) == before


def test_a_removal_that_fails_partway_puts_everything_back(monkeypatch, data_dir):
    packages = data_dir / "packages"
    _two_add_ons(packages)
    before = _tree(packages)
    real_replace = os.replace
    moves = []

    def replace(src, dst):
        moves.append(src)
        if len(moves) == 3:
            raise PermissionError(13, "The process cannot access the file", src)
        real_replace(src, dst)

    monkeypatch.setattr(package_store.os, "replace", replace)

    with pytest.raises(RuntimeError, match="Everything that was there is back"):
        package_store.remove(["edge-tts"])

    assert _tree(packages) == before
    assert _leftovers(data_dir) == []


_KILL_REMOVAL = """\
import os, sys
from opencut import helpers
from opencut.core import package_store

helpers.OPENCUT_DIR = sys.argv[1]
real_replace = os.replace
calls = []

def replace(src, dst):
    calls.append(dst)
    if len(calls) == 3:  # two entries are aside, then the power goes
        os._exit(9)
    real_replace(src, dst)

os.replace = replace
package_store.remove(["edge-tts"])
"""


def test_a_removal_killed_partway_is_put_back_by_the_next_store_change(data_dir, tmp_path):
    packages = data_dir / "packages"
    _two_add_ons(packages)
    before = _tree(packages)
    script = tmp_path / "kill_removal.py"
    script.write_text(_KILL_REMOVAL, encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")])))

    killed = subprocess.run(
        [sys.executable, str(script), str(data_dir)],
        env=env, capture_output=True, text=True, timeout=120, creationflags=NO_WINDOW,
    )

    assert killed.returncode == 9, killed.stderr
    assert _tree(packages) != before
    assert package_store.remove(["not-installed"]) == []
    assert _tree(packages) == before
    assert not [p for p in data_dir.iterdir() if p.name.startswith(".packages-replaced-")]


def test_a_removal_waits_for_an_install_then_says_so(another_install, monkeypatch, data_dir):
    packages = data_dir / "packages"
    _two_add_ons(packages)
    before = _tree(packages)
    monkeypatch.setattr(package_store.store_lock.__wrapped__, "__defaults__", (package_store._log, 0.3))

    with pytest.raises(package_store.StoreBusy):
        package_store.remove(["edge-tts"])

    assert _tree(packages) == before


def test_a_reinstall_keeps_the_mark_of_a_project_someone_installed_for_itself(monkeypatch, data_dir):
    packages = data_dir / "packages"
    _dist(packages, "shared", "1.0", {"shared/__init__.py": b"shared 1.0"}, requested=True)

    def stage(target):
        _dist(target, "new-tool", "1.0", {"new_tool/__init__.py": b"new"}, requires=["shared>=1"], requested=True)
        _dist(target, "shared", "1.5", {"shared/__init__.py": b"shared 1.5"})

    _pip(monkeypatch, stage)
    package_store.install(["new-tool"], keep_shared=False)

    assert (packages / "shared-1.5.dist-info" / "REQUESTED").is_file()
    assert package_store.remove(["new-tool"]) == ["new-tool 1.0"]
    assert (packages / "shared" / "__init__.py").read_bytes() == b"shared 1.5"


def test_the_server_removes_a_tool_from_the_command_line(data_dir):
    packages = data_dir / "packages"
    _two_add_ons(packages)
    env = dict(
        os.environ,
        OPENCUT_HOME=str(data_dir),
        PYTHONPATH=os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")])),
    )

    result = subprocess.run(
        [sys.executable, "-m", "opencut.server", "--remove-tools", "edge-tts"],
        env=env, capture_output=True, text=True, encoding="utf-8", timeout=300, creationflags=NO_WINDOW,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "[ok] edge-tts is removed. Restart OpenCut if it's running." in result.stdout.splitlines()
    assert sorted(package_store.read_dists(str(packages))) == ["certifi", "mediapipe", "numpy"]
