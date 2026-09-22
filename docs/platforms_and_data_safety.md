# Platform setup and preserving existing data

Use this with the [raw-to-RF guide](../README.md). It covers the operating
system of the **Python kernel or MATLAB process**, not the computer displaying
the notebook. A Windows browser connected to a Linux Jupyter server still
runs Linux code and uses Linux paths.

## 1. Choose the runtime and paths

### Linux and macOS

Keep the current SQLite connection in `Utils/session_edits.py`.
It selects `unix-dotfile` locking for the lab's CIFS-mounted data.
Use the Bash examples and `.venv/bin/python` in the main guide.

Change the data root to the actual mounted location on that machine. A
Linux `/mnt/...` path is not automatically the same location as a macOS
`/Volumes/...` path. Do not copy a Python virtual environment between
operating systems; install the dependencies in an environment on the target
machine. Sorting still requires its supported CUDA runtime.

### Windows with a Linux kernel

For SSH to a Linux server, run the main guide's commands in that remote
terminal. Keep the current database connection.

For WSL, use Linux paths and a Linux environment inside WSL. A Windows drive
such as `D:` is normally accessed as `/mnt/d` there. Native Windows
Python, WSL Python, and a remote Linux kernel have different paths and
environments. Do not mix their interpreter paths or database connections.
Keep one active database owner; WSL does not make simultaneous access from
native Windows and Linux safe.

### Native Windows: notebook and reader setup

The following configures the RF Python environment and notebook examples in
**PowerShell**. Replace every example root with a real directory. It does not
establish support for the separate sorting repository's Linux launchers.

```powershell
$env:RFMAP_CODE_DIR = "C:\code\rfmapping"
$env:RF_MATLAB_DIR = Join-Path $env:RFMAP_CODE_DIR "matlab"
$env:RFMAP_WORK_DIR = "D:\rf_work"
$env:RECORDING_ROOT = "D:\recordings"
$env:RF_MOUSE = "mouse_01"
$env:RF_DATE = "260918"
$env:RF_SESSION = "2"
$env:RF_SESSIONS = "2"
$env:RF_PROBES = "A"
$env:RF_MOUSE_DIR = Join-Path $env:RECORDING_ROOT $env:RF_MOUSE
$env:RF_SESSION_DIR = Join-Path (Join-Path $env:RF_MOUSE_DIR $env:RF_DATE) "$($env:RF_DATE)_$($env:RF_SESSION)"

# Skip environment creation if you already have a suitable environment.
py -3.12 -m venv "$env:RFMAP_CODE_DIR\.venv"
$env:RFMAP_PYTHON = "$env:RFMAP_CODE_DIR\.venv\Scripts\python.exe"
& $env:RFMAP_PYTHON -m pip install -e "$($env:RFMAP_CODE_DIR)[analysis]" jupyterlab ipykernel
& $env:RFMAP_PYTHON -m ipykernel install --user --name rfmapping --display-name "RF mapping"
Set-Location $env:RFMAP_CODE_DIR
& $env:RFMAP_PYTHON -m jupyterlab
```

Select the **RF mapping** kernel. The main guide's Python cells that use
`os.environ` now read these Windows settings.

In a notebook parameter cell, either use the environment variable or a
Windows path:

```python
import os
base_dir = os.environ["RF_MOUSE_DIR"]
# Equivalent examples:
# base_dir = r"D:\recordings\mouse_01"
# base_dir = "D:/recordings/mouse_01"
```

Here `base_dir` is the mouse directory; the timing notebooks append the
date and session. A Python raw string cannot end with a single backslash.
Keep the same directory nesting and filename capitalization as the Linux
dataset.

Bash `export`, `<<'PY'` heredocs, `rsync`, and `bin/python`
commands are not PowerShell commands. For a multi-line Python example from a
Bash block, copy its Python body into a notebook cell or a separate script and
run it with the selected interpreter. Use the documented Linux environment for
the upstream sorting stage unless that repository provides a working Windows
setup.

Environment variables must be set before starting Jupyter or MATLAB. A running
Jupyter server retains its old environment even when its kernel is restarted.
When changing sessions or paths, update the derived variables too and relaunch
the server from the configured terminal. The same applies to an existing
MATLAB process using `getenv`.

### Native Windows: change the SQLite connection before manual timing

In [`Utils/session_edits.py`](../Utils/session_edits.py), find
`SessionEditStore._connect`. The current connection is:

```python
database_uri = f"{database_path.as_uri()}?vfs=unix-dotfile"
database_connection = sqlite3.connect(database_uri, uri=True, timeout=30)
```

For **native Windows only**, replace those lines with:

```python
database_uri = database_path.as_uri()
database_connection = sqlite3.connect(database_uri, uri=True, timeout=30)
```

Leave the remainder of the function, including the transaction and close
handling, unchanged. Keep `as_uri()`; it correctly encodes spaces and
`#` in paths. Restart the kernel and rerun the imports after changing
your Windows copy.

Removing the explicit VFS selects the operating system's default; native
Windows normally uses `win32`. Keep the original `unix-dotfile`
line for the Linux/macOS lab setup. This fixes the connection selection, not
every dependency or upstream launcher. Native Windows execution has not been
validated here. [SQLite VFS documentation](https://www.sqlite.org/vfs.html).

The database file itself is portable between Windows and Unix-like systems.
Do not create a new database merely because the original was created on a
different OS. [SQLite file portability](https://www.sqlite.org/onefile.html).

## 2. Preserve and check the existing database

### Locate the correct file before running the main cell

The manual notebook opens:

```text
<mouse directory>/session_slices.sqlite3
```

Despite its filename, it uses the current `session_edits` table.
There is one row per `(date, session_id)`. Mouse identity is scoped by
the containing directory; it is not stored as a column in the current table.

- Use the database belonging to this mouse.
- Do not substitute another mouse's database because the filenames match.
- Do not copy one existing database over another. This replaces history;
  it does not merge the records.
- Do not delete or recreate a database to solve a path, permission, Windows VFS,
  or schema error.
- When expecting saved corrections, confirm that the file exists **before**
  calling `check_session_edits()`. That helper creates a database if
  the selected path is missing. A typo can therefore leave you with an empty
  database at the wrong location.

The opener reports a legacy ordered schema instead of automatically converting
it. Keep the original and arrange an explicit, backed-up migration that
preserves sequential index meanings. The current code does not supply a
general migration command. `Could not initialize session-edit database`
can also reflect a connection failure; it does not mean the existing file
should be reset.

### Make a closed backup that cannot replace an existing file

1. Stop notebook kernels, analysis jobs, and database browsers that use this
   database on **all** computers.
2. Finish/close active database work before copying. Do not copy a live database
   or delete journal, WAL, or lock files to make an error disappear. If those
   files remain after a failed session, resolve recovery with the original
   runtime before treating a single-file copy as a complete backup.
3. Choose a new backup filename, or a new local destination folder for a
   Windows working copy.
4. Run the following in a separate Python session, changing both paths:

```python
from pathlib import Path
import shutil

source = Path("D:/recordings/mouse_01/session_slices.sqlite3")
destination = Path("D:/recordings/mouse_01/session_slices.before_windows_copy.sqlite3")

with source.open("rb") as original, destination.open("xb") as backup:
    shutil.copyfileobj(original, backup)

print("Preserved:", destination)
```

Linux/macOS users substitute their own source and destination paths.
`"xb"` refuses to open an existing destination and raises
`FileExistsError`; choose another name instead of changing it to
`"wb"`. A failed or interrupted copy is not a verified backup.

Keep the source database unchanged during transfer. Use a local working copy
on native Windows and preserve its mouse-directory layout. If both copies
subsequently acquire different corrections, compare the affected sessions
before reconciling them; copying a whole file over the other discards its rows.

Do not open the same shared database concurrently from native Windows and
Linux/macOS, or from a database browser using different locking. Different
VFS locking methods may not see each other's locks. A network drive is not a
database synchronization service. [SQLite locking compatibility](https://www.sqlite.org/vfs.html).

### Know which manual actions change saved records

The main cell in [`matlab.ipynb`](../matlab.ipynb) can save newly
derived corrections when a session has no saved record. It then writes
`data/on_list_times.npy` after its trial-count check. The
`is_overwrite_parameters` variable does not guard either action.

| Action | Effect |
| --- | --- |
| `getSessionInfo()`, `getDelete()`, `getInterp()` | Read the selected session's saved correction |
| `deleteByFrame()` / `deleteByRange()` | Append delete indices to the stored record |
| `interp()` | Set or replace the session's one interpolation |
| `clearInterps()` | Remove its saved interpolation |
| Store-level `replace_edits()` | Replace the session record; an empty edit list removes it |

These mutations do not ask for confirmation. Delete indices refer to the
original detector output; interpolation indices refer to the sequence after
deletion. Preserve the database and existing timing arrays before revising
those decisions. Do not replay historical edit cells on a different session.

## 3. Other path and cache assumptions

| Location | What to check on another computer or rerun |
| --- | --- |
| `matlab.ipynb` / `matlab_auto.ipynb` | Change the mouse-level `base_dir` and recording identity. The notebooks expect the repeated date nesting described in the README. |
| `matlab_auto.ipynb` export | Keep `is_save_on_list_time=False` during preview. `True` replaces the same onset file as the manual notebook; the auto route does not apply the SQLite corrections. |
| Session metadata | Cached paths can still refer to the original machine after a copy. Read the current structure.oebin when selecting a new data root; do not reset the correction database to fix a path. |
| `load_rfmap.ipynb` | Set the intended RF source directory and filename. Loading is separate from optional RF-result persistence. |
| `Utils.rfmap` / `Utils.rf_cache` | A matching `result_path` cache is reused. A cache-key mismatch can replace that `.npz` with a recalculated result; choose different filenames for results you want to retain. Omitting `result_path` keeps the result in memory. |
| RF input filenames | Preserve the capitalization of the sorting and spike-time directories exactly as shown in the main guide. Case-sensitive filesystems distinguish uppercase and lowercase names. |
| MATLAB `FindInInterval` | A compiled MEX file is platform-specific. Run `mexext`, check `which FindInInterval -all`, and use or compile a binary for the active MATLAB platform. Moving a Linux binary to Windows/macOS does not make it compatible. |
| Runtime directories | Recreate Python environments on the target OS. Do not treat copied `__pycache__`, Numba caches, or compiled binaries as portable analysis inputs. |
