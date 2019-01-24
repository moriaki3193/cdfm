# -*- coding: utf-8 -*-
"""Package profiling script
"""
from line_profiler import LineProfiler
# from cdfm.config import DTYPE
# from cdfm.models import equations as Eqn


if __name__ == '__main__':
    pr = LineProfiler()
    pr.enable()
    pr.disable()
    pr.print_stats()
