#!/usr/bin/env python
"""
batch_progress.py -- tiny always-on-top progress window for run_multiple.py.

Polls a JSON status file (written by run_multiple.py after each session) and
shows:  X / Y sessions done,  average per session,  estimated time remaining,
elapsed, and the session currently running. Pure standard library (tkinter +
json), so it adds no dependencies and is safe to close or kill at any time.

run_multiple.py launches this for you; you normally never call it by hand:
    python batch_progress.py --status <path-to-_progress.json> [--poll 1.0]
"""
import argparse
import json
import time

import tkinter as tk
from tkinter import ttk


def fmt_dur(seconds):
    """Seconds -> '1h 02m' / '3m 12s' / '45s' / '--'."""
    if seconds is None or seconds < 0:
        return "--"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return "%dh %02dm" % (h, m)
    if m:
        return "%dm %02ds" % (m, s)
    return "%ds" % s


class ProgressWindow:
    def __init__(self, status_path, poll):
        self.status_path = status_path
        self.poll_ms = max(200, int(poll * 1000))

        self.root = tk.Tk()
        self.root.title("MPS batch progress")
        self.root.resizable(False, False)
        try:
            self.root.attributes("-topmost", True)
        except Exception:
            pass

        self.var_done = tk.StringVar(value="waiting for first session...")
        self.var_avg = tk.StringVar(value="avg per session:  --")
        self.var_eta = tk.StringVar(value="est. remaining:   --")
        self.var_elapsed = tk.StringVar(value="elapsed:          --")
        self.var_current = tk.StringVar(value="")

        tk.Label(self.root, textvariable=self.var_done,
                 font=("Segoe UI", 15, "bold")).grid(row=0, column=0, padx=14, pady=(12, 4), sticky="w")
        self.bar = ttk.Progressbar(self.root, orient="horizontal", length=320, mode="determinate")
        self.bar.grid(row=1, column=0, padx=14, pady=4)
        tk.Label(self.root, textvariable=self.var_avg,
                 font=("Consolas", 10)).grid(row=2, column=0, padx=14, pady=1, sticky="w")
        tk.Label(self.root, textvariable=self.var_eta,
                 font=("Consolas", 11, "bold")).grid(row=3, column=0, padx=14, pady=1, sticky="w")
        tk.Label(self.root, textvariable=self.var_elapsed,
                 font=("Consolas", 10)).grid(row=4, column=0, padx=14, pady=1, sticky="w")
        tk.Label(self.root, textvariable=self.var_current,
                 font=("Segoe UI", 9), fg="#555").grid(row=5, column=0, padx=14, pady=(4, 8), sticky="w")
        ttk.Button(self.root, text="Close", command=self.root.destroy).grid(row=6, column=0, pady=(0, 10))

        self.root.after(100, self.tick)

    def _read(self):
        try:
            with open(self.status_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def tick(self):
        st = self._read()
        if st:
            total = int(st.get("total", 0) or 0)
            done = int(st.get("done", 0) or 0)
            durations = st.get("durations") or []
            start_time = st.get("start_time")
            finished = bool(st.get("finished"))
            current = st.get("current")
            current_start = st.get("current_start")

            self.bar["maximum"] = max(total, 1)
            self.bar["value"] = min(done, total)
            self.var_done.set("%d / %d sessions done" % (done, total))

            avg = (sum(durations) / len(durations)) if durations else None
            self.var_avg.set("avg per session:  %s" % fmt_dur(avg))

            remaining = max(total - done, 0)
            if finished or remaining == 0:
                self.var_eta.set("est. remaining:   done ✓")
            else:
                eta = avg * remaining if avg is not None else None
                # discount the time already spent on the in-flight session
                if eta is not None and current_start:
                    eta = max(eta - max(time.time() - current_start, 0.0), 0.0)
                self.var_eta.set("est. remaining:   %s   (%d to go)" % (fmt_dur(eta), remaining))

            if start_time:
                self.var_elapsed.set("elapsed:          %s" % fmt_dur(time.time() - start_time))

            if finished:
                self.var_current.set("batch complete")
                self.root.title("MPS batch -- done")
            elif current:
                run_for = (time.time() - current_start) if current_start else None
                self.var_current.set("running: %s   (%s)" % (current, fmt_dur(run_for)))
            else:
                self.var_current.set("")

        self.root.after(self.poll_ms, self.tick)

    def run(self):
        self.root.mainloop()


def main():
    ap = argparse.ArgumentParser(description="Live progress window for run_multiple.py")
    ap.add_argument("--status", required=True, help="Path to the _progress.json written by run_multiple.py")
    ap.add_argument("--poll", type=float, default=1.0, help="Seconds between refreshes (default 1.0)")
    args = ap.parse_args()
    ProgressWindow(args.status, args.poll).run()


if __name__ == "__main__":
    main()
