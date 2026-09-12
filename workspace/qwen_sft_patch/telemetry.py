"""Sample process RSS and device-wide NVML counters; missing sensors remain null."""
import threading
import time
from sft_core import append_jsonl

class Telemetry:
    def __init__(self, path, gpu_uuid, interval=1.0):
        self.path, self.interval = path, interval
        self.phase = "setup"
        self.stop_event = threading.Event()
        self.samples = []
        self.errors = {}
        self.nv = self.handle = self.proc = None
        try:
            import psutil
            self.proc = psutil.Process()
        except Exception as e:
            self.errors["rss"] = repr(e)
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nv = pynvml
            # Torch UUID identifies the actual visible GPU even when CUDA_VISIBLE_DEVICES remaps it.
            self.handle = pynvml.nvmlDeviceGetHandleByUUID(str(gpu_uuid))
        except Exception as e:
            self.errors["nvml"] = repr(e)
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.thread.start()

    def _sample(self):
        row = {"monotonic_s": time.perf_counter(), "phase": self.phase,
               "rss_bytes": None, "device_memory_bytes": None, "gpu_utilization_pct": None, "power_w": None}
        if self.proc:
            try:
                row["rss_bytes"] = self.proc.memory_info().rss
            except Exception as e:
                self.errors["rss"] = repr(e)
        if self.handle is not None:
            for key, fn in [("device_memory_bytes", lambda: self.nv.nvmlDeviceGetMemoryInfo(self.handle).used),
                            ("gpu_utilization_pct", lambda: self.nv.nvmlDeviceGetUtilizationRates(self.handle).gpu),
                            ("power_w", lambda: self.nv.nvmlDeviceGetPowerUsage(self.handle)/1000)]:
                try:
                    row[key] = fn()
                except Exception as e:
                    self.errors[key] = repr(e)
        self.samples.append(row)
        append_jsonl(self.path, row)

    def _loop(self):
        while not self.stop_event.is_set():
            self._sample()
            self.stop_event.wait(self.interval)

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
        self._sample()
        if self.nv:
            self.nv.nvmlShutdown()
        out = {"sensor_errors": self.errors, "sample_interval_seconds": self.interval,
               "monitor_scope": "sampling begins after early file preflight and ends after final save; all means all sampled phases, not every second of job wall time",
               "nvml_scope": "entire selected GPU; includes other processes and idle power; no baseline subtraction"}
        for phase in ["all", "model_load", "train", "checkpoint", "final_save"]:
            rows = self.samples if phase == "all" else [r for r in self.samples if r["phase"] == phase]
            result = {}
            for key in ["rss_bytes", "device_memory_bytes"]:
                vals = [r[key] for r in rows if r[key] is not None]
                result["peak_sampled_"+key] = max(vals) if vals else None
            energy = duration = util_integral = util_duration = 0.0
            for a,b in zip(self.samples, self.samples[1:]):
                delta = b["monotonic_s"]-a["monotonic_s"]
                if delta <= 0 or delta > 3*self.interval or (phase != "all" and (a["phase"] != phase or b["phase"] != phase)):
                    continue
                if a["power_w"] is not None and b["power_w"] is not None:
                    energy += delta*(a["power_w"]+b["power_w"])/2
                    duration += delta
                if a["gpu_utilization_pct"] is not None and b["gpu_utilization_pct"] is not None:
                    util_integral += delta*(a["gpu_utilization_pct"]+b["gpu_utilization_pct"])/2
                    util_duration += delta
            result.update({"sampled_energy_j": energy if duration else None,
                           "energy_covered_seconds": duration,
                           "mean_gpu_utilization_pct": util_integral/util_duration if util_duration else None})
            out[phase] = result
        return out
