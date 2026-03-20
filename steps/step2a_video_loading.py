import tkinter as tk
from tkinter import ttk
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
import sys
import importlib
from pathlib import Path
import re


class Step2aVideoLoading(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.processing_complete = False

        # Canvas + scrollbar for entire frame
        self.canvas = tk.Canvas(self)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Title
        ttk.Label(
            self.scrollable_frame,
            text="Step 2a: Video Loading and Chunking",
            font=("Arial", 16, "bold")
        ).grid(row=0, column=0, columnspan=2, pady=20, padx=10, sticky="w")

        ttk.Label(
            self.scrollable_frame,
            text="Loads miniscope videos with OpenCV frame-by-frame reading and chunked zarr writes. "
                 "Handles corrupt frames gracefully without hanging.",
            wraplength=800
        ).grid(row=1, column=0, columnspan=2, pady=10, padx=10, sticky="w")

        # ── Left panel: controls ──────────────────────────────────────────────
        self.control_frame = ttk.LabelFrame(self.scrollable_frame, text="Loading Parameters")
        self.control_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

        # File pattern
        ttk.Label(self.control_frame, text="File Pattern:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.pattern_var = tk.StringVar(value=r".*\.avi$")
        ttk.Entry(self.control_frame, textvariable=self.pattern_var, width=30).grid(
            row=0, column=1, padx=10, pady=10, sticky="w")

        def open_regex101():
            import webbrowser
            webbrowser.open_new("https://regex101.com/")

        regex_help = ttk.Label(self.control_frame, text="?", foreground="blue", cursor="hand2")
        regex_help.grid(row=0, column=2, padx=5, pady=10, sticky="w")
        regex_help.bind("<Button-1>", lambda e: open_regex101())

        # Downsampling
        ttk.Label(self.control_frame, text="Downsampling:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.downsample_frame = ttk.LabelFrame(self.control_frame, text="Downsample Factors")
        self.downsample_frame.grid(row=1, column=1, padx=10, pady=5, sticky="w")

        ttk.Label(self.downsample_frame, text="Frame:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.frame_ds_var = tk.IntVar(value=1)
        ttk.Entry(self.downsample_frame, textvariable=self.frame_ds_var, width=5).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(self.downsample_frame, text="Height:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.height_ds_var = tk.IntVar(value=1)
        ttk.Entry(self.downsample_frame, textvariable=self.height_ds_var, width=5).grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(self.downsample_frame, text="Width:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.width_ds_var = tk.IntVar(value=1)
        ttk.Entry(self.downsample_frame, textvariable=self.width_ds_var, width=5).grid(row=2, column=1, padx=5, pady=5)

        # Downsample strategy (kept for UI compat)
        ttk.Label(self.control_frame, text="Downsample Strategy:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.ds_strategy_var = tk.StringVar(value="subset")
        self.ds_strategy_combo = ttk.Combobox(self.control_frame, textvariable=self.ds_strategy_var, width=15)
        self.ds_strategy_combo['values'] = ('subset', 'mean')
        self.ds_strategy_combo.grid(row=2, column=1, padx=10, pady=10, sticky="w")

        # Line splitting
        ttk.Label(self.control_frame, text="Line Splitting Detection:").grid(row=3, column=0, padx=10, pady=10, sticky="w")
        self.line_splitting_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            self.control_frame,
            text="Detect and remove line splitting frames",
            variable=self.line_splitting_var
        ).grid(row=3, column=1, padx=10, pady=10, sticky="w")

        # Run button
        self.run_button = ttk.Button(self.control_frame, text="Load Videos", command=self.run_loading)
        self.run_button.grid(row=4, column=0, columnspan=2, pady=20, padx=10)

        # Status + progress
        self.status_var = tk.StringVar(value="Ready to load videos")
        ttk.Label(self.control_frame, textvariable=self.status_var).grid(row=5, column=0, columnspan=2, pady=10)
        self.progress = ttk.Progressbar(self.control_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.grid(row=6, column=0, columnspan=2, pady=10, padx=10, sticky="ew")

        # ── Right panel: log ──────────────────────────────────────────────────
        self.log_frame = ttk.LabelFrame(self.scrollable_frame, text="Processing Log")
        self.log_frame.grid(row=2, column=1, padx=10, pady=10, sticky="nsew")
        log_scroll = ttk.Scrollbar(self.log_frame)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text = tk.Text(self.log_frame, height=20, width=50, yscrollcommand=log_scroll.set)
        self.log_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        log_scroll.config(command=self.log_text.yview)

        # ── Bottom: preview ───────────────────────────────────────────────────
        self.viz_frame = ttk.LabelFrame(self.scrollable_frame, text="Video Preview")
        self.viz_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        self.fig = plt.Figure(figsize=(8, 4), dpi=100)
        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas_fig.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.scrollable_frame.grid_columnconfigure(0, weight=1)
        self.scrollable_frame.grid_columnconfigure(1, weight=3)
        self.scrollable_frame.grid_rowconfigure(2, weight=3)
        self.scrollable_frame.grid_rowconfigure(3, weight=2)

        self.controller.register_step_button('Step2aVideoLoading', self.run_button)

    # ── UI helpers ────────────────────────────────────────────────────────────

    def log(self, message):
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.update_idletasks()

    def update_progress(self, value):
        self.progress["value"] = value
        self.update_idletasks()

    def run_loading(self):
        if not self.controller.state.get('initialized', False):
            self.status_var.set("Error: Please complete Step 1 initialization first")
            self.log("Error: Please complete Step 1 initialization first")
            return

        self.status_var.set("Loading videos...")
        self.progress["value"] = 0
        self.log("Starting video loading process...")

        ds_frame = self.frame_ds_var.get()
        ds_h = self.height_ds_var.get()
        ds_w = self.width_ds_var.get()
        for label, val in [("frame", ds_frame), ("height", ds_h), ("width", ds_w)]:
            if val <= 0:
                self.status_var.set(f"Error: {label} downsample factor must be positive")
                self.log(f"Error: {label} downsample factor must be positive")
                return

        param_load_videos = {
            "pattern": self.pattern_var.get(),
            "dtype": np.uint8,
            "downsample": {"frame": ds_frame, "height": ds_h, "width": ds_w},
            "downsample_strategy": self.ds_strategy_var.get(),
            "cache_path": self.controller.state.get('cache_path', ''),
        }

        thread = threading.Thread(
            target=self._load_videos_thread,
            args=(
                self.controller.state.get('input_dir', ''),
                param_load_videos,
                100,
                self.controller.state.get('cache_path', ''),
                self.line_splitting_var.get(),
            )
        )
        thread.daemon = True
        thread.start()

    def on_show_frame(self):
        params = self.controller.get_step_parameters('Step2aVideoLoading')
        if params:
            if 'pattern' in params:
                self.pattern_var.set(params['pattern'])
            if 'downsample' in params and isinstance(params['downsample'], dict):
                self.frame_ds_var.set(params['downsample'].get('frame', 1))
                self.height_ds_var.set(params['downsample'].get('height', 1))
                self.width_ds_var.set(params['downsample'].get('width', 1))
            if 'downsample_strategy' in params:
                self.ds_strategy_var.set(params['downsample_strategy'])
            self.log("Parameters loaded from file")

    # ── Core loading thread ───────────────────────────────────────────────────

    def _load_videos_thread(self, input_dir, param_load_videos, video_percent, cache_path, detect_line_splitting):
        """
        Loads videos using OpenCV frame-by-frame reading with chunked zarr writes.

        Why cv2 instead of ffmpeg pipe:
          - cv2.VideoCapture.read() returns (False, None) on corrupt frames instead
            of hanging indefinitely — no watchdog or restart logic needed.
          - Same chunked zarr write strategy as before: O(1) RAM per chunk regardless
            of file size.
          - ffmpeg is still used for robust probing (frame count, fps, duration).

        Strategy:
          1. Probe each file with ffmpeg for accurate metadata.
          2. Open each file with cv2.VideoCapture.
          3. Read frame-by-frame; apply temporal + spatial downsampling in Python.
          4. Buffer frames in chunks of ~10 min; write each chunk to zarr before
             reading the next (peak RAM = one chunk, not the whole video).
          5. Corrupt/unreadable frames get a blank substitute and a warning log.
          6. Save the completed zarr as an xarray DataArray for downstream steps.
        """
        import os, sys, shutil, subprocess
        from pathlib import Path

        # ── Conda env / PATH fix ──────────────────────────────────────────────
        env_root = Path(sys.executable).parent
        candidates = [
            env_root / "Library" / "bin",
            env_root / "Library" / "usr" / "bin",
            env_root / "Scripts",
            env_root / "bin",
        ]
        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        prepend = [str(p) for p in candidates if p.exists() and str(p) not in path_parts]
        if prepend:
            os.environ["PATH"] = os.pathsep.join(prepend + path_parts)
        os.environ["CONDA_PREFIX"] = str(env_root)
        os.environ["CONDA_DEFAULT_ENV"] = env_root.name

        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            guess = env_root / "Library" / "bin" / "ffmpeg.exe"
            if guess.exists():
                ffmpeg_path = str(guess)
        if not ffmpeg_path:
            self.log("ERROR: Could not resolve ffmpeg.")
            self.status_var.set("Error: FFmpeg not found")
            return
        self.log(f"Using ffmpeg at: {ffmpeg_path}")
        os.environ["FFMPEG_BINARY"] = ffmpeg_path

        try:
            module_base_path = Path(__file__).parent.parent
            if str(module_base_path) not in sys.path:
                sys.path.append(str(module_base_path))

            # saving_utilities
            utilities_spec = importlib.util.find_spec("saving_utilities")
            if utilities_spec:
                saving_utilities = importlib.import_module("saving_utilities")
                get_optimal_chk = saving_utilities.get_optimal_chk
                save_files = saving_utilities.save_files
                self.log("Imported saving_utilities")
            else:
                self.log("ERROR: saving_utilities module not found")
                self.status_var.set("Error: saving_utilities missing")
                return

            import cv2
            import xarray as xr
            import dask
            import dask.array as da
            from dask.distributed import Client, LocalCluster
            from natsort import natsorted
            import ffmpeg
            import zarr
            import psutil

            self.log("All modules imported successfully")

            # ── Pure probe function (ffmpeg for accuracy) ─────────────────────

            def probe_video(fname):
                """
                Robust frame-count probe via ffmpeg.
                Trusts duration*fps over nb_frames header when they disagree
                by more than 5% (common in AVIs written by DAQ software).
                """
                probe = ffmpeg.probe(fname)
                vs = next((s for s in probe["streams"] if s["codec_type"] == "video"), None)
                if vs is None:
                    return None

                w = int(vs["width"])
                h = int(vs["height"])

                fps = 10.0
                for rate_key in ("r_frame_rate", "avg_frame_rate"):
                    fps_str = vs.get(rate_key, "")
                    if fps_str and fps_str not in ("", "0/0"):
                        try:
                            num, den = map(int, fps_str.split("/"))
                            if den > 0:
                                fps = num / den
                                break
                        except Exception:
                            pass

                duration = 0.0
                for src in (vs, probe.get("format", {})):
                    try:
                        duration = float(src.get("duration") or 0)
                        if duration > 0:
                            break
                    except Exception:
                        pass

                nb_raw = vs.get("nb_frames", "")
                nb_from_header = int(nb_raw) if (nb_raw and nb_raw not in ("", "N/A")) else 0
                nb_from_duration = int(round(duration * fps))

                if nb_from_header == 0:
                    f = nb_from_duration
                elif nb_from_duration > 0:
                    discrepancy = abs(nb_from_header - nb_from_duration) / max(nb_from_header, 1)
                    f = nb_from_duration if discrepancy > 0.05 else nb_from_header
                else:
                    f = nb_from_header

                return {"width": w, "height": h, "frames": f, "fps": fps, "duration": duration}

            def detect_line_splitting_frames_fn(xarray_data):
                left_edge = xarray_data.isel(width=slice(0, 20))
                left_edge_means = left_edge.mean(dim=["height", "width"]).compute()
                overall_mean = left_edge_means.mean().item()
                overall_std = left_edge_means.std().item()
                threshold = overall_mean + 2 * overall_std
                return np.where(left_edge_means > threshold)[0].tolist()

            # ── Dask cluster ──────────────────────────────────────────────────
            self.log("Initializing Dask cluster...")
            n_workers = self.controller.state.get("n_workers", 8)
            memory_limit = self.controller.state.get("memory_limit", "200GB")

            cluster = LocalCluster(
                n_workers=n_workers,
                memory_limit=memory_limit,
                resources={"MEM": 1},
                threads_per_worker=2,
                dashboard_address=":8787",
            )
            client = Client(cluster)
            self.log(f"Dask Dashboard: {client.dashboard_link}")
            self.controller.state["dask_dashboard_url"] = client.dashboard_link

            def show_dashboard_popup():
                popup = tk.Toplevel(self.controller)
                popup.title("Dask Dashboard Ready")
                popup.geometry("400x150")
                popup.attributes("-topmost", True)
                ttk.Label(popup, text="Dask dashboard is now available:", wraplength=380).pack(pady=(10, 5))
                ttk.Label(popup, text=client.dashboard_link).pack(pady=5)
                def copy_link():
                    popup.clipboard_clear()
                    popup.clipboard_append(client.dashboard_link)
                    copy_btn.config(text="Copied!")
                copy_btn = ttk.Button(popup, text="Copy Link", command=copy_link)
                copy_btn.pack(pady=5)
                ttk.Button(popup, text="OK", command=popup.destroy).pack(pady=10)
                try:
                    popup.bell()
                except Exception:
                    pass

            self.controller.after(100, show_dashboard_popup)
            self.update_progress(10)

            # ── Parse downsampling params ─────────────────────────────────────
            ds = param_load_videos.get("downsample", {"frame": 1, "height": 1, "width": 1})
            ds_frame = max(1, ds.get("frame", 1))
            ds_h = max(1, ds.get("height", 1))
            ds_w = max(1, ds.get("width", 1))

            # ── Find video files ──────────────────────────────────────────────
            self.log(f"Scanning {input_dir} for pattern: {param_load_videos['pattern']}")
            vpath = os.path.normpath(input_dir)
            if os.path.isfile(vpath):
                vlist = [vpath]
            else:
                vlist = natsorted([
                    os.path.join(vpath, v) for v in os.listdir(vpath)
                    if re.search(param_load_videos["pattern"], v)
                ])

            if not vlist:
                self.log("No videos found matching pattern!")
                self.status_var.set("Error: No videos found")
                return
            self.log(f"Found {len(vlist)} video file(s)")

            # Validate ffmpeg binary
            try:
                subprocess.run([ffmpeg_path, "-version"], capture_output=True, text=True, check=True)
                self.log("FFmpeg binary validated OK")
            except Exception as e:
                self.log(f"ERROR: FFmpeg binary check failed: {e}")
                self.status_var.set("Error: FFmpeg not working")
                return

            self.update_progress(15)

            # ── Probe all files ───────────────────────────────────────────────
            corrupted_files = []
            valid_files = []  # list of (fname, info, out_h, out_w)

            for fname in vlist:
                fsize_mb = os.path.getsize(fname) / (1024 ** 2)
                self.log(f"\nProbing: {os.path.basename(fname)} ({fsize_mb:.1f} MB)")
                try:
                    info = probe_video(fname)
                    if info is None:
                        raise ValueError("No video stream found in file")

                    orig_h, orig_w = info["height"], info["width"]
                    out_h = orig_h // ds_h
                    out_w = orig_w // ds_w
                    approx_out_frames = info["frames"] // ds_frame

                    raw_mb = info["frames"] * orig_h * orig_w / (1024 ** 2)
                    out_mb = approx_out_frames * out_h * out_w / (1024 ** 2)

                    self.log(
                        f"  Source: {orig_w}x{orig_h}, {info['frames']} frames, "
                        f"{info['fps']:.2f} fps, {info['duration']:.1f}s  (~{raw_mb:.0f} MB raw)"
                    )
                    self.log(
                        f"  Output: {out_w}x{out_h}, ~{approx_out_frames} frames  (~{out_mb:.0f} MB)"
                        f"  [{ds_frame}x temporal, {ds_h}x{ds_w} spatial — applied in Python]"
                    )

                    valid_files.append((fname, info, out_h, out_w))

                except Exception as e:
                    self.log(f"  FAILED to probe: {e}")
                    corrupted_files.append({
                        "filename": os.path.basename(fname),
                        "full_path": fname,
                        "error": str(e),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    })

            if not valid_files:
                self.log("No valid videos to load!")
                self.status_var.set("Error: No valid videos")
                return

            # ── Pre-run cleanup + permission checks ───────────────────────────
            self.log(f"\n{'='*60}")
            self.log("Pre-run checks...")

            ram_total_gb = psutil.virtual_memory().total / (1024**3)
            ram_avail_gb = psutil.virtual_memory().available / (1024**3)

            combined_zarr_path = os.path.join(cache_path, "step2a_stream_tmp.zarr")
            total_expected = sum(info["frames"] // max(ds_frame, 1) for _, info, _, _ in valid_files)

            _, first_info, first_out_h, first_out_w = valid_files[0]
            out_fps = first_info["fps"] / max(ds_frame, 1)
            frames_per_chunk = max(1, int(10 * 60 * out_fps))
            chunk_mb = frames_per_chunk * first_out_h * first_out_w / (1024**2)
            total_output_mb = total_expected * first_out_h * first_out_w / (1024**2)
            n_chunks = max(1, total_expected // frames_per_chunk)

            self.log(f"System RAM: {ram_avail_gb:.1f} GB available / {ram_total_gb:.1f} GB total")
            self.log(f"Files to load:       {len(valid_files)}")
            self.log(f"Output frames:       {total_expected:,}  (~{out_fps:.1f} fps after temporal ds)")
            self.log(f"Output resolution:   {first_out_w}x{first_out_h} px")
            self.log(f"Total output size:   {total_output_mb/1024:.2f} GB")
            self.log(f"Chunk size:          {frames_per_chunk:,} frames / 10 min  ({chunk_mb:.0f} MB per chunk in RAM)")
            self.log(f"Expected chunks:     ~{n_chunks} chunks total")
            self.log(f"Reader:              OpenCV (cv2.VideoCapture) — corrupt frames skipped gracefully")
            self.log(f"Writing to zarr:     {combined_zarr_path}")

            _files_to_clear = [
                combined_zarr_path,
                os.path.join(cache_path, "step2a_chunking_info.json"),
                os.path.join(cache_path, "line_splitting_frames.txt"),
                os.path.join(cache_path, "frame_index_map.txt"),
                os.path.join(cache_path, "corrupted_files.json"),
            ]

            for _fp in _files_to_clear:
                if os.path.exists(_fp):
                    try:
                        if os.path.isdir(_fp):
                            import shutil as _shu
                            _shu.rmtree(_fp)
                        else:
                            os.remove(_fp)
                        self.log(f"  Cleared: {os.path.basename(_fp)}")
                    except Exception as _e:
                        self.log(f"  WARNING: Could not clear {os.path.basename(_fp)}: {_e}")

            _test_file = os.path.join(cache_path, "_write_test.tmp")
            try:
                with open(_test_file, 'w') as _tf:
                    _tf.write("test")
                os.remove(_test_file)
                self.log(f"  Write permission OK: {cache_path}")
            except Exception as _e:
                self.log(f"  ERROR: Cannot write to cache dir: {_e}")
                self.status_var.set("Error: No write permission to cache dir")
                return

            self.log("Pre-run checks complete\n")

            # ── Create root zarr store ────────────────────────────────────────
            z = zarr.open(
                combined_zarr_path,
                mode="w",
                shape=(0, first_out_h, first_out_w),
                chunks=(min(frames_per_chunk, 500), first_out_h, first_out_w),
                dtype=np.uint8,
            )

            global_written = 0
            t_start = time.time()

            # ── Load each file with cv2 ───────────────────────────────────────
            for file_idx, (fname, info, out_h, out_w) in enumerate(valid_files):
                file_label = os.path.basename(fname)
                file_expected = info["frames"] // max(ds_frame, 1)
                orig_h = info["height"]
                orig_w = info["width"]

                self.log(f"\nFile {file_idx+1}/{len(valid_files)}: {file_label}")
                self.log(
                    f"  Expected: {file_expected:,} frames  |  "
                    f"Duration: {info['duration']:.0f}s  |  "
                    f"Size: {os.path.getsize(fname)/(1024**3):.2f} GB"
                )

                cap = cv2.VideoCapture(fname)
                if not cap.isOpened():
                    self.log(f"  ERROR: cv2 could not open {file_label} — skipping")
                    corrupted_files.append({
                        "filename": file_label,
                        "full_path": fname,
                        "error": "cv2.VideoCapture could not open file",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    })
                    continue

                chunk_buf = []
                file_written = 0
                corrupt_count = 0
                raw_frame_counter = 0  # counts every frame read from the file
                t_file = time.time()

                try:
                    while True:
                        ret, frame_raw = cap.read()

                        # End of file (or unrecoverable read error)
                        if not ret:
                            if file_written + len(chunk_buf) < file_expected * 0.95:
                                # Ended significantly early — log a warning
                                self.log(
                                    f"  WARNING: cv2 read ended at {file_written + len(chunk_buf):,} frames "
                                    f"(expected ~{file_expected:,}). "
                                    f"File may be truncated or have a large corrupt tail."
                                )
                            break

                        raw_frame_counter += 1

                        # Temporal downsampling: keep frame 1, ds_frame+1, 2*ds_frame+1, ...
                        if ds_frame > 1 and (raw_frame_counter % ds_frame) != 1:
                            continue

                        # Convert to grayscale if needed
                        if frame_raw is None or frame_raw.size == 0:
                            # Rare: ret=True but frame is empty — substitute blank
                            frame_gray = np.zeros((orig_h, orig_w), dtype=np.uint8)
                            corrupt_count += 1
                        elif len(frame_raw.shape) == 3:
                            frame_gray = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2GRAY)
                        else:
                            frame_gray = frame_raw

                        # Spatial downsampling (subset — numpy slice, zero copy)
                        if ds_h > 1 or ds_w > 1:
                            frame_out = frame_gray[::ds_h, ::ds_w].copy()
                        else:
                            frame_out = frame_gray.copy()

                        chunk_buf.append(frame_out)

                        # Flush chunk to zarr
                        if len(chunk_buf) >= frames_per_chunk:
                            block = np.stack(chunk_buf)
                            new_global = global_written + block.shape[0]
                            z.resize(new_global, out_h, out_w)
                            z[global_written:new_global] = block
                            global_written = new_global
                            file_written += block.shape[0]
                            chunk_buf.clear()

                            elapsed = time.time() - t_start
                            fps_actual = global_written / max(elapsed, 1)
                            remaining = (total_expected - global_written) / max(fps_actual, 1)
                            file_pct = file_written / max(file_expected, 1) * 100
                            total_pct = global_written / max(total_expected, 1) * 100
                            ram_used = psutil.virtual_memory().percent
                            self.log(
                                f"  File {file_pct:.0f}%  |  Overall {global_written:,}/{total_expected:,} "
                                f"({total_pct:.0f}%)  |  {fps_actual:.0f} fps  |  "
                                f"ETA {remaining/60:.0f} min  |  RAM {ram_used:.0f}%"
                                + (f"  | corrupt frames so far: {corrupt_count}" if corrupt_count else "")
                            )
                            self.update_progress(15 + total_pct * 0.60)

                    # Flush remainder
                    if chunk_buf:
                        block = np.stack(chunk_buf)
                        new_global = global_written + block.shape[0]
                        z.resize(new_global, out_h, out_w)
                        z[global_written:new_global] = block
                        global_written = new_global
                        file_written += block.shape[0]
                        chunk_buf.clear()

                finally:
                    cap.release()

                t_file_elapsed = time.time() - t_file
                self.log(
                    f"  File done: {file_written:,} frames in {t_file_elapsed/60:.1f} min "
                    f"({file_written/max(t_file_elapsed,1):.0f} fps avg)"
                    + (f"  | {corrupt_count} corrupt/blank frame(s) substituted" if corrupt_count else "")
                )

            self.update_progress(75)

            # ── Corrupted files log ───────────────────────────────────────────
            if corrupted_files:
                import json
                self.log(f"\n{len(corrupted_files)} file(s) could not be opened")
                try:
                    with open(os.path.join(cache_path, "corrupted_files.json"), "w") as f:
                        json.dump(corrupted_files, f, indent=4)
                except Exception as e:
                    self.log(f"Could not save corrupted_files.json: {e}")
                output_dir = self.controller.state.get("dataset_output_path", "")
                if output_dir:
                    try:
                        with open(os.path.join(output_dir, "corrupted_files.txt"), "w") as f:
                            f.write(f"Total corrupted files: {len(corrupted_files)}\n{'='*60}\n\n")
                            for cf in corrupted_files:
                                f.write(
                                    f"File: {cf['filename']}\nPath: {cf['full_path']}\n"
                                    f"Error: {cf['error']}\nTimestamp: {cf['timestamp']}\n\n"
                                )
                    except Exception as e:
                        self.log(f"Could not save corrupted_files.txt: {e}")

            if global_written == 0:
                self.log("No frames written — aborting.")
                self.status_var.set("Error: Loading failed")
                return

            # ── Open completed zarr as lazy dask array ────────────────────────
            z_read = zarr.open(combined_zarr_path, mode="r")
            combined_da = da.from_zarr(z_read)
            self.log(f"\nZarr complete: {global_written:,} frames at {first_out_w}x{first_out_h}")

            # video_percent subset
            total_frames = combined_da.shape[0]
            frames_to_use = int(total_frames * (video_percent / 100))
            if frames_to_use < total_frames:
                self.log(f"Subsetting to {frames_to_use}/{total_frames} frames ({video_percent}%)")
                combined_da = combined_da[:frames_to_use]

            step2a_varr = xr.DataArray(
                combined_da,
                dims=["frame", "height", "width"],
                coords={
                    "frame": np.arange(combined_da.shape[0]),
                    "height": np.arange(combined_da.shape[1]),
                    "width": np.arange(combined_da.shape[2]),
                },
            ).rename("fluorescence")

            self.log(f"DataArray: shape={step2a_varr.shape}, dtype={step2a_varr.dtype}")
            self.update_progress(65)

            # ── Line splitting detection ──────────────────────────────────────
            line_splitting_frames = []
            if detect_line_splitting:
                self.log("Detecting line splitting frames...")
                try:
                    line_splitting_frames = detect_line_splitting_frames_fn(step2a_varr)

                    if line_splitting_frames:
                        self.log(f"Found {len(line_splitting_frames)} line splitting frames")
                        try:
                            with open(os.path.join(cache_path, "line_splitting_frames.txt"), "w") as f:
                                f.write(str(line_splitting_frames))
                        except Exception as e:
                            self.log(f"Error saving line_splitting_frames.txt: {e}")

                        original_shape = step2a_varr.shape
                        all_frames = np.arange(step2a_varr.sizes["frame"])
                        frames_to_keep = np.setdiff1d(all_frames, line_splitting_frames)
                        step2a_varr = step2a_varr.isel(frame=frames_to_keep)
                        step2a_varr = step2a_varr.assign_coords(frame=np.arange(step2a_varr.shape[0]))

                        frame_index_map = frames_to_keep
                        try:
                            with open(os.path.join(cache_path, "frame_index_map.txt"), "w") as f:
                                f.write("# Maps post-linesplit frame index -> original video frame index\n")
                                f.write(f"Total original frames: {total_frames}\n")
                                f.write(f"Frames after removal: {len(frame_index_map)}\n")
                                f.write(f"Original indices kept: {frame_index_map.tolist()}")
                        except Exception as e:
                            self.log(f"Error saving frame_index_map.txt: {e}")

                        self.controller.state.setdefault("results", {}).setdefault("step2a", {})["frame_index_map"] = frame_index_map
                        self.log(f"Removed line splitting frames: {original_shape} → {step2a_varr.shape}")
                    else:
                        self.log("No line splitting frames detected")
                        try:
                            with open(os.path.join(cache_path, "line_splitting_frames.txt"), "w") as f:
                                f.write("[]")
                        except Exception as e:
                            self.log(f"Error saving line_splitting_frames.txt: {e}")

                except Exception as e:
                    self.log(f"Error during line splitting detection: {e}")

            # ── Chunking + save ───────────────────────────────────────────────
            self.log("\nComputing optimal chunking...")
            step2a_chk, _ = get_optimal_chk(step2a_varr, dtype=float)
            self.log(f"Optimal chunk: {step2a_chk}")

            try:
                import json
                with open(os.path.join(cache_path, "step2a_chunking_info.json"), "w") as f:
                    json.dump(step2a_chk, f, indent=4)
            except Exception as e:
                self.log(f"Could not save chunking info: {e}")

            self.update_progress(75)

            self.log("Saving to zarr cache (Step 2b will read this lazily)...")
            chunked_arr = step2a_varr.chunk(
                {"frame": step2a_chk["frame"], "height": -1, "width": -1}
            ).rename("step2a_varr")

            try:
                step2a_varr = save_files(chunked_arr, cache_path, overwrite=True)
                self.log("Saved successfully")
            except Exception as e:
                self.log(f"Error saving: {e}")
                self.status_var.set(f"Error saving: {e}")
                return

            self.update_progress(90)

            # ── Preview ───────────────────────────────────────────────────────
            self.log("Creating preview...")
            self.after_idle(lambda: self.create_preview(step2a_varr))

            # ── Store results ─────────────────────────────────────────────────
            self.controller.state.setdefault("results", {})["step2a"] = {
                "step2a_varr": step2a_varr,
                "step2a_chk": step2a_chk,
                "line_splitting_frames": line_splitting_frames,
            }

            self.after_idle(lambda: self.status_var.set("Video loading complete"))
            self.log("\nVideo loading complete!")
            if line_splitting_frames:
                self.log(f"({len(line_splitting_frames)} line splitting frames removed)")
            self.update_progress(100)
            self.controller.status_var.set("Videos loaded successfully")
            self.processing_complete = True
            self.controller.after(0, lambda: self.controller.on_step_complete(self.__class__.__name__))

        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.log(f"Error: {str(e)}")
            self.log(f"Traceback: {sys.exc_info()}")
            if self.controller.state.get("autorun_stop_on_error", True):
                self.controller.autorun_enabled = False
                self.controller.autorun_indicator.config(text="")

    # ── Preview ───────────────────────────────────────────────────────────────

    def create_preview(self, step2a_varr):
        try:
            self.fig.clear()
            axs = self.fig.subplots(2, 2)

            frame = step2a_varr.isel(frame=0).compute()
            im1 = axs[0, 0].imshow(frame, cmap="gray")
            axs[0, 0].set_title("First Frame")
            self.fig.colorbar(im1, ax=axs[0, 0])

            axs[0, 1].hist(frame.values.flatten(), bins=50)
            axs[0, 1].set_title("Intensity Distribution")
            axs[0, 1].set_xlabel("Pixel Value")
            axs[0, 1].set_ylabel("Count")

            mean_frame = step2a_varr.isel(frame=slice(0, 100)).mean("frame").compute()
            im2 = axs[1, 0].imshow(mean_frame, cmap="gray")
            axs[1, 0].set_title("Mean Frame (first 100)")
            self.fig.colorbar(im2, ax=axs[1, 0])

            axs[1, 1].axis("off")
            axs[1, 1].text(
                0.1, 0.5,
                (
                    f"Array Information:\n"
                    f"Shape: {step2a_varr.shape}\n"
                    f"Data Type: {step2a_varr.dtype}\n"
                    f"Dimensions: {step2a_varr.dims}\n"
                    f"Chunks: {step2a_varr.chunks}\n"
                ),
                transform=axs[1, 1].transAxes,
                verticalalignment="center",
            )

            self.fig.tight_layout()
            self.canvas_fig.draw()

        except Exception as e:
            self.log(f"Error creating preview: {str(e)}")