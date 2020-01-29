from time import time_ns
import numpy as np
from collections import deque


def time_mapping(granularity):
    return {
        Timer.MINUTES: "min",
        Timer.SECONDS: "s",
        Timer.MILLISECONDS: "ms",
        Timer.MICROSECONDS: "μs",
        Timer.NANOSECONDS: "ns",
    }[granularity]


class Timer:
    MINUTES = int(60e9)
    SECONDS = int(1e9)
    MILLISECONDS = int(1e6)
    MICROSECONDS = int(1e3)
    NANOSECONDS = 1

    @staticmethod
    def print_nothing(time):
        pass

    @classmethod
    def on_iter(cls, iterable, **kwargs):
        timer = cls(**kwargs)
        timer.start()
        for v in iterable:
            timer.tick()
            yield v

    @staticmethod
    def print_from_format_string(format_string, granularity=None):
        if granularity is None:
            granularity = Timer.MILLISECONDS
        return lambda time: print(
            format_string.format(time, Timer.time_mapping(granularity))
        )

    @staticmethod
    def moving_average(print_func, size=10, frequency=1):
        window = deque()
        idx = 0

        def new_print_func(ms):
            nonlocal idx
            idx += 1
            window.append(ms)
            if len(window) >= size:
                window.popleft()
            if idx % frequency == 0:
                return print_func(np.mean(list(window)))

        return new_print_func

    @staticmethod
    def time_mapping(granularity):
        return {
            Timer.MINUTES: "min",
            Timer.SECONDS: "s",
            Timer.MILLISECONDS: "ms",
            Timer.MICROSECONDS: "μs",
            Timer.NANOSECONDS: "ns",
        }[granularity]

    def __init__(
        self, granularity=None, print_func=None, disable=False, if_in_profile=False
    ):
        self.granularity = (
            granularity if granularity is not None else Timer.MILLISECONDS
        )
        self.interval_string = self.time_mapping(self.granularity)
        self._start = None

        if if_in_profile:
            self.disable = _STACK_COUNT == 0
        else:
            self.disable = disable

        format_string = "time: {:.3f}{}"
        self.print_func = (
            print_func
            if print_func is not None
            else self.print_from_format_string(format_string, self.granularity)
        )

    def __enter__(self):
        self._start = time_ns()
        return self

    def start(self):
        self.__enter__()

    def tick(self, print_func=None):
        now = time_ns()
        interval = None
        if print_func is None:
            print_func = self.print_func
        if self._start is not None:
            interval = self._tock(print_func)
        self._start = now
        return interval

    def _tock(self, print_func, now=None):
        if now is None:
            now = time_ns()
        interval = float(now - self._start) / self.granularity
        if not self.disable:
            print_func(interval)
        return interval

    def __exit__(self, type, value, traceback):
        self._tock(self.print_func)
