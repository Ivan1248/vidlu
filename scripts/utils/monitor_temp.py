import os
import re
import time
import argparse
import json
from datetime import datetime

from tqdm import trange, tqdm
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Experiment running script')
parser.add_argument("--duration", type=float, default=60)
parser.add_argument("--cpu_count", type=int, default=4)
parser.add_argument("--period", type=float, default=0.1)
parser.add_argument("--prefix", type=str, default="temp_log")

args = parser.parse_args()


class Stopwatch:
    """A stopwatch that can be used as a context manager.

    Example:
        with Stopwatch as sw:
            sleep(1)
            assert sw.running and sw.time >= 1.
        assert not sw.running and sw.time >= 1.

    Example:
        sw = Stopwatch().start()
        sleep(1)
        assert sw.running and sw.time >= 1.
        sw.stop()
        assert not sw.running and sw.time >= 1.

        sw.reset()
        assert not sw.running and sw.time == sw.start_time == 0
    """
    __slots__ = '_time_func', 'start_time', '_time', 'running'

    def __init__(self, time_func=time.time):
        self._time_func = time_func
        self.reset()

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.stop()

    def __str__(self):
        return f"Stopwatch(time={self.time})"

    @property
    def time(self):
        return self._time + self._time_func() - self.start_time if self.running else self._time

    def reset(self):
        self._time = 0.
        self.start_time = None
        self.running = False
        return self

    def start(self):
        if self.running:
            raise RuntimeError("Stopwatch is already running.")
        self.start_time = self._time_func()
        self.running = True
        return self

    def stop(self):
        if self.running:
            self._time = self.time
            self.running = False
        return self._time


def run_command(cmd):
    with os.popen(cmd) as stream:
        return stream.read()


def get_temps_and_freqs(cpu_count=1, pattern=r"\d+(?:\.\d+)?"):
    temps = run_command("sensors | grep Tdie;").split("\n")[:-1]
    freqs = [
        run_command(f"sudo cat /sys/devices/system/cpu/cpu{i}/cpufreq/cpuinfo_cur_freq").strip()
        for i in range(cpu_count)]
    return [float(re.findall(pattern, line)[0]) for line in temps], \
           [int(line.strip()) for line in freqs]


def get_utilization():
    lines = run_command("nvidia-smi | grep %;").split("\n")[:-1]
    return [float(re.findall(r"\d+%", line)[-1][:-1]) for line in lines]


log_path = f"{args.prefix}_s{datetime.now().strftime('%H_%M')}_c{args.duration}_p{args.period}.json"

times, tempses, freqses, gpuutses = [], [], [], []

with Stopwatch() as t:
    pbar = tqdm(total=int(args.duration * 10))
    while t.time < args.duration:
        time.sleep(0.1)
        curr_temps, curr_freqs = get_temps_and_freqs(cpu_count=args.cpu_count)
        times.append(t.time)
        tempses.append(curr_temps)
        freqses.append(curr_freqs)
        gpuutses.append(get_utilization())
        pbar.update(int(t.time * 10) - pbar.n)

print(f"max: {np.max(np.array(tempses))}")
print(f"mean: {np.average(np.array(tempses))}")
print(f"std: {np.std(np.array(tempses))}")
with open(log_path, "w") as f:
    json.dump(dict(tempses=tempses, freqses=freqses), f)

tempses = np.transpose(np.array(tempses))
freqses = np.transpose(np.array(freqses))
gpuutses = np.transpose(np.array(gpuutses))

fig, axs = plt.subplots(3, 2, gridspec_kw={'width_ratios': [3, 1]})
axs[1, 1].remove()
axs[2, 1].remove()

for temps in tempses:
    for freqs in freqses:
        axs[0, 1].scatter(temps, freqs, c='black', alpha=max([10 / freqses.size, 0.01]), edgecolors='none')
axs[0, 1].set_xlabel("temp.")
axs[0, 1].set_ylabel("freq.")

for temps in tempses:
    axs[0, 0].plot(times, temps)
axs[0, 0].set_xlabel("t")
axs[0, 0].set_ylabel("temp.")

for freqs in freqses:
    axs[1, 0].plot(times, freqs, alpha=1 / args.cpu_count)
axs[1, 0].plot(times, freqses.mean(0), c='black')
axs[1, 0].set_xlabel("t")
axs[1, 0].set_ylabel("freq.")

for gpuuts in gpuutses:
    axs[2, 0].plot(times, gpuuts, alpha=1 / len(gpuutses))
axs[2, 0].plot(times, gpuutses.mean(0), c='black')
axs[2, 0].set_xlabel("t")
axs[2, 0].set_ylabel("GPU util.")

plt.tight_layout()
plt.show()
