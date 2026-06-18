#!/usr/bin/env python
"""
run_multiple.py -- batch-run run_step.py over many sessions, one at a time.

For each session it launches run_step.py as its OWN subprocess, so every
session gets a fresh Dask cluster + Tk root that are fully torn down when that
session finishes (or crashes) before the next one starts. A session that fails
-- even a hard native crash -- is just a non-zero exit code here: it's logged
and the batch moves on to the next session.

Each session reads its own  cache_data/processing_parameters.json  (run_step
does that automatically), so you don't pass parameters here.

The combined terminal output of every session is streamed live AND saved to a
per-session log file under the batch log folder. Two cross-session reports are
also written into that folder and refreshed after every session:
  failures.log                 every failed step: session, step, error message
  component_counts_table.txt   all sessions x steps, component count in each cell

===========================================================================
QUICK START
===========================================================================
  # Every *_Processed session under a root folder, full pipeline each:
  python run_multiple.py --root D:\\BE_Processed_first_20

  # Explicit sessions, in this order:
  python run_multiple.py D:\\BE_Processed_first_20\\3334_17_Processed ^
                         D:\\BE_Processed_first_20\\3340_5_Processed

  # Re-derive the CNMF chain from cache instead of reprocessing from raw:
  python run_multiple.py --root D:\\BE_Processed_first_20 --run-args "--from 3b --to 8c --qc"

  # Put logs on the data drive, and skip a session that hangs > 6 hours:
  python run_multiple.py --root D:\\BE_Processed_first_20 ^
                         --log-dir D:\\BE_Processed_first_20\\batch_logs --timeout-min 360

By default each session is run as:
    run_step.py --all --no-dask --keep-going --qc --results-dir <session>
Change that with --run-args (do NOT put --results-dir there; it's added per
session). This script only orchestrates; it needs no scientific packages, so
run it with any Python -- it launches run_step.py inside the local ./env.
"""

import os
import re
import sys
import glob
import time
import shlex
import signal
import argparse
import threading
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
RUN_STEP = os.path.join(_HERE, "run_step.py")
ENV_DIR = os.path.join(_HERE, "env")

# What each session is run as, unless overridden with --run-args.
DEFAULT_RUN_ARGS = "--all --no-dask --keep-going --qc"


def _say(msg):
    print(msg)
    try:
        sys.stdout.flush()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Session discovery
# ---------------------------------------------------------------------------

def session_label(session_dir):
    """'.../3334_17_Processed' -> '3334_17'."""
    base = os.path.basename(os.path.normpath(session_dir))
    if base.endswith("_Processed"):
        base = base[:-len("_Processed")]
    return base


def _sort_key(session_dir):
    """Sort by (animal, session) numerically when possible."""
    bits = session_label(session_dir).split("_")
    out = []
    for b in bits[:2]:
        out.append((0, int(b)) if b.isdigit() else (1, b))
    return out


def find_sessions(root):
    """Every '*_Processed' folder under root that contains a cache_data."""
    root = os.path.abspath(os.path.expanduser(root))
    found = [d for d in glob.glob(os.path.join(root, "*_Processed"))
             if os.path.isdir(os.path.join(d, "cache_data"))]
    return sorted(found, key=_sort_key)


# ---------------------------------------------------------------------------
# Cross-session reports: failures.log + component_counts_table.txt
# ---------------------------------------------------------------------------

# Pipeline order, used to order the columns of the component-count table.
_STEP_ORDER = ["1", "2a", "2b", "2c", "2d", "2e", "2f", "3a", "3b", "3c",
               "4a", "4b", "4c", "4d", "4e", "4f", "4g", "4h", "4hq", "5a", "5b",
               "6a", "6b", "6c", "6d", "6e", "7a", "7b", "7c", "7d", "7e", "7f",
               "8a", "8b", "8c"]

# A QC summary line looks like:  "  step 4c : 925 neurons  align=..."
_RE_QC_COUNT = re.compile(r"^\s*step\s+(\S+)\s*:\s*(\d+)\s+neurons")
# A failed step looks like:      "[run_step] step 4e INCOMPLETE  (1.8s)"
_RE_STEP_FAIL = re.compile(r"\[run_step\] step (\S+) (?:INCOMPLETE|FAILED)")
_RE_ERRORISH = re.compile(r"(Error:|ERROR|Traceback|Exception)")
_RE_TS = re.compile(r"^\[\d\d:\d\d:\d\d\]\s*")


def scan_session_log(log_path):
    """Pull (failed-steps-with-errors, component-counts-per-step) out of one
    session's captured run_step output.

    Returns (fails, counts) where fails is a list of (step_id, error_message)
    and counts is {step_id: n_components} from the per-step QC summary lines."""
    fails, counts, recent_err = [], {}, ""
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                mc = _RE_QC_COUNT.match(line)
                if mc:
                    counts[mc.group(1)] = int(mc.group(2))
                    continue
                stripped = _RE_TS.sub("", line.strip())
                if _RE_ERRORISH.search(stripped) and "conda.cli" not in stripped:
                    recent_err = stripped          # remember the most recent real error
                mf = _RE_STEP_FAIL.search(line)
                if mf:
                    fails.append((mf.group(1), recent_err))
                    recent_err = ""
    except Exception:
        pass
    return fails, counts


def read_cumulative_counts(session_dir):
    """Read a session's CUMULATIVE component counts from
    cache_data/qc_plots/component_counts.txt, which qc_cnmf maintains across all
    runs. This is what lets a partial-range batch (e.g. --from 5a) still produce
    a table that includes the earlier steps' counts. Returns {step_id: n}."""
    path = os.path.join(session_dir, "cache_data", "qc_plots", "component_counts.txt")
    counts = {}
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = re.match(r"\s*step\s+(\S+)\s*:\s*(\d+)", line)
                if m:
                    counts[m.group(1)] = int(m.group(2))
    except Exception:
        pass
    return counts


def write_failures_log(run_dir, fail_rows):
    """fail_rows: list of (session_label, step_id, error_message)."""
    path = os.path.join(run_dir, "failures.log")
    lines = ["FAILED STEPS  (updated %s)" % time.strftime("%Y-%m-%d %H:%M:%S"),
             "%-12s %-5s %s" % ("session", "step", "error"),
             "-" * 78]
    if not fail_rows:
        lines.append("(no failures so far)")
    for label, step, err in fail_rows:
        lines.append("%-12s %-5s %s" % (label, step, err or "(no error message captured)"))
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except Exception as exc:
        _say("[run_multiple] could not write failures.log: %s" % exc)


def write_component_table(run_dir, count_rows):
    """count_rows: list of (session_label, {step_id: n_components})."""
    path = os.path.join(run_dir, "component_counts_table.txt")
    seen = set()
    for _label, counts in count_rows:
        seen.update(counts)
    cols = [s for s in _STEP_ORDER if s in seen] + sorted(s for s in seen if s not in _STEP_ORDER)
    header = "%-12s " % "session" + " ".join("%6s" % c for c in cols)
    lines = ["COMPONENT COUNTS PER STEP  (updated %s)" % time.strftime("%Y-%m-%d %H:%M:%S"),
             "(count after each QC'd step; blank = step not run or not QC'd)",
             "",
             header,
             "-" * len(header)]
    for label, counts in count_rows:
        cells = " ".join("%6s" % (counts[c] if c in counts else "") for c in cols)
        lines.append("%-12s %s" % (label, cells))
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except Exception as exc:
        _say("[run_multiple] could not write component table: %s" % exc)


# ---------------------------------------------------------------------------
# Launching run_step.py inside the local conda env
# ---------------------------------------------------------------------------

def build_launcher():
    """Command prefix that runs a python script inside ./env.

    `conda run -p ./env` is preferred because it sets up ONLY this env even if
    another conda env (e.g. base) is currently active -- launching env's
    python.exe directly while base is active can load the wrong DLLs and crash
    numpy on Windows. Falls back to the env interpreter, then the current one.
    """
    try:
        import shutil
        conda = shutil.which("conda")
    except Exception:
        conda = None
    if conda:
        return [conda, "run", "--no-capture-output", "-p", ENV_DIR, "python"]
    direct = (os.path.join(ENV_DIR, "python.exe") if os.name == "nt"
              else os.path.join(ENV_DIR, "bin", "python"))
    if os.path.exists(direct):
        return [direct]
    return [sys.executable]


def _tee(stream, logf):
    """Pump a child's output to both the console and a log file, line by line."""
    try:
        for line in iter(stream.readline, ""):
            if line == "":
                break
            try:
                sys.stdout.write(line)
                sys.stdout.flush()
            except Exception:
                pass
            try:
                logf.write(line)
                logf.flush()
            except Exception:
                pass
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _kill_tree(proc):
    """Kill the child and any processes it spawned (e.g. Dask workers)."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def run_session(session_dir, run_args, log_path, timeout_sec):
    """Run one session as a subprocess. Returns (status, returncode, seconds)."""
    cmd = build_launcher() + [RUN_STEP] + list(run_args) + ["--results-dir", session_dir]

    popen_kwargs = dict(stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        bufsize=1, universal_newlines=True)
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True   # own process group for clean kill

    logf = open(log_path, "w", encoding="utf-8")
    t0 = time.time()
    header = ("===========================================================\n"
              "SESSION : %s\n"
              "STARTED : %s\n"
              "COMMAND : %s\n"
              "===========================================================\n"
              % (session_dir, time.strftime("%Y-%m-%d %H:%M:%S"), " ".join(cmd)))
    _say(header.rstrip())
    logf.write(header)
    logf.flush()

    try:
        proc = subprocess.Popen(cmd, **popen_kwargs)
    except Exception as exc:
        msg = "[run_multiple] FAILED to launch run_step: %s\n" % exc
        _say(msg.rstrip())
        logf.write(msg)
        logf.close()
        return ("LAUNCH-ERROR", None, time.time() - t0)

    reader = threading.Thread(target=_tee, args=(proc.stdout, logf))
    reader.daemon = True
    reader.start()

    status, rc = "OK", None
    try:
        rc = proc.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        _say("[run_multiple] session exceeded %s s -- killing it." % timeout_sec)
        _kill_tree(proc)
        try:
            rc = proc.wait(timeout=15)
        except Exception:
            pass
        status = "TIMEOUT"
    finally:
        reader.join(timeout=5)

    dt = time.time() - t0
    if status != "TIMEOUT":
        status = "OK" if rc == 0 else "FAILED"

    footer = ("\n[run_multiple] %s -> %s  (exit=%s, %.1f min)\n"
              % (session_label(session_dir), status, rc, dt / 60.0))
    _say(footer.rstrip())
    logf.write(footer)
    logf.close()
    return (status, rc, dt)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        prog="run_multiple.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Run run_step.py over many sessions sequentially "
                    "(see top of file for examples).",
    )
    p.add_argument("sessions", nargs="*",
                   help="Session *_Processed folders to run (or use --root).")
    p.add_argument("--root",
                   help="Run every *_Processed session under this folder.")
    p.add_argument("--run-args", default=DEFAULT_RUN_ARGS,
                   help="Args passed to run_step.py for each session "
                        "(default: %r). Do NOT include --results-dir." % DEFAULT_RUN_ARGS)
    p.add_argument("--log-dir",
                   help="Where to write per-session logs (default: ./batch_logs).")
    p.add_argument("--timeout-min", type=float, default=None,
                   help="Kill a session (and its workers) after N minutes; continue.")
    p.add_argument("--start-at", metavar="LABEL",
                   help="Skip sessions until this label (e.g. 3340_5) to resume a batch.")
    p.add_argument("--list", action="store_true",
                   help="List the sessions that would run, then exit.")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.root:
        sessions = find_sessions(args.root)
    else:
        sessions = [os.path.abspath(os.path.expanduser(s)) for s in args.sessions]

    if not sessions:
        _say("[run_multiple] No sessions found. Give *_Processed folders, or --root DIR.")
        return 2

    if args.start_at:
        labels = [session_label(s) for s in sessions]
        if args.start_at in labels:
            sessions = sessions[labels.index(args.start_at):]
        else:
            _say("[run_multiple] --start-at %r not among sessions; running all." % args.start_at)

    if args.list:
        _say("Would run %d session(s):" % len(sessions))
        for i, s in enumerate(sessions, 1):
            _say("  %2d. %-12s %s" % (i, session_label(s), s))
        _say("\nPer-session command: run_step.py %s --results-dir <session>" % args.run_args)
        return 0

    run_args = shlex.split(args.run_args, posix=(os.name != "nt"))
    timeout_sec = args.timeout_min * 60.0 if args.timeout_min else None

    log_root = os.path.abspath(os.path.expanduser(args.log_dir)) if args.log_dir \
        else os.path.join(os.getcwd(), "batch_logs")
    run_dir = os.path.join(log_root, "batch_" + time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    _say("=" * 70)
    _say("BATCH: %d session(s)" % len(sessions))
    _say("per-session: run_step.py %s --results-dir <session>" % args.run_args)
    _say("logs: %s" % run_dir)
    _say("=" * 70)

    summary = []
    count_rows = []   # (label, {step: n_components})  for the component table
    fail_rows = []    # (label, step, error)           for failures.log
    for i, sd in enumerate(sessions, 1):
        label = session_label(sd)
        _say("\n" + "#" * 70)
        _say("# [%d/%d] %s" % (i, len(sessions), label))
        _say("#" * 70)
        log_path = os.path.join(run_dir, "%02d_%s.log" % (i, label))
        if not os.path.isdir(sd):
            _say("[run_multiple] not a directory, skipping: %s" % sd)
            summary.append((label, "MISSING", None, 0.0, sd))
            fail_rows.append((label, "-", "session folder missing: %s" % sd))
            count_rows.append((label, {}))
            write_failures_log(run_dir, fail_rows)
            write_component_table(run_dir, count_rows)
            continue
        status, rc, dt = run_session(sd, run_args, log_path, timeout_sec)
        summary.append((label, status, rc, dt, log_path))

        # ----- refresh the two cross-session reports after every session -----
        s_fails, s_counts = scan_session_log(log_path)
        # Merge in the session's CUMULATIVE counts (from qc_plots) so the table
        # spans the whole pipeline even when this batch ran only part of it
        # (e.g. --from 5a). This batch's freshly scraped log values take
        # precedence for the steps it just ran.
        merged_counts = read_cumulative_counts(sd)
        merged_counts.update(s_counts)
        count_rows.append((label, merged_counts))
        if s_fails:
            for step, err in s_fails:
                fail_rows.append((label, step, err))
        elif status != "OK":
            fail_rows.append((label, "-", "session %s (exit=%s)" % (status, rc)))
        write_failures_log(run_dir, fail_rows)
        write_component_table(run_dir, count_rows)

    # ----- final summary (console + file) -----------------------------------
    lines = []
    lines.append("=" * 70)
    lines.append("BATCH SUMMARY  (%s)" % time.strftime("%Y-%m-%d %H:%M:%S"))
    lines.append("%-14s %-10s %-7s %-8s %s" % ("session", "status", "exit", "minutes", "log"))
    lines.append("-" * 70)
    n_ok = 0
    for label, status, rc, dt, log_path in summary:
        if status == "OK":
            n_ok += 1
        lines.append("%-14s %-10s %-7s %-8.1f %s"
                     % (label, status, "" if rc is None else rc, dt / 60.0,
                        os.path.basename(log_path) if log_path else ""))
    lines.append("-" * 70)
    lines.append("%d/%d OK,  %d need attention" % (n_ok, len(summary), len(summary) - n_ok))
    lines.append("reports: failures.log,  component_counts_table.txt")
    lines.append("=" * 70)
    report = "\n".join(lines)
    _say("\n" + report)
    try:
        with open(os.path.join(run_dir, "_summary.txt"), "w", encoding="utf-8") as f:
            f.write(report + "\n")
    except Exception:
        pass

    # final refresh of the reports (in case the last session was MISSING etc.)
    write_failures_log(run_dir, fail_rows)
    write_component_table(run_dir, count_rows)

    # echo the component-counts table to the console too, so a partial-range
    # batch (e.g. --run-args "--from 4h --to 4h ...") still prints the table
    # when it finishes -- not just saves it to the log folder.
    try:
        with open(os.path.join(run_dir, "component_counts_table.txt"),
                  encoding="utf-8") as f:
            _say("\n" + f.read())
    except Exception as exc:
        _say("[run_multiple] could not print component table: %s" % exc)

    return 0 if n_ok == len(summary) else 1


if __name__ == "__main__":
    sys.exit(main())
