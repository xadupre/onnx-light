"""
.. _l-example-parallel-for-profiling:

Inspect ParallelFor profiling from Python
=========================================

Parallel-region profiling is disabled by default. This example enables it with
a bounded collector, runs an ``Abs`` model, and inspects an immutable report.
Passing ``hardware_counters=True`` explicitly requests the Linux grouped
``perf_event_open`` backend. Linux may require ``CAP_PERFMON`` or a permissive
``kernel.perf_event_paranoid`` setting. The backend rejects rather than scales
multiplexed groups, so derived metrics exist only when ``time_running`` equals
``time_enabled``. It also marks multi-thread regions unsuitable because a
thread-bound perf group cannot represent their aggregate work. Unsupported
platforms and denied counters retain portable timing; ``None`` is distinct from
a valid zero.
"""

# sphinx_gallery_thumbnail_path = "_static/gallery_thumbnails/parallel_for_profiling.png"

from __future__ import annotations

import numpy

from onnx_light.onnx import TensorProto
from onnx_light.onnx_lib import parser
from onnx_light.onnx_py import _onnxpykernels

runtime = _onnxpykernels.runtime

model = parser.parse_model(
    '<ir_version: 10, opset_import: ["" : 18]>\n'
    "agraph (float[N] x) => (float[N] y) {\n"
    "  y = Abs(x)\n"
    "}\n"
)
values = -numpy.ones(100_000, dtype=numpy.float32)
tensor_proto = TensorProto()
tensor_proto.name = "x"
tensor_proto.dims.append(values.size)
tensor_proto.data_type = int(TensorProto.FLOAT)
tensor_proto.raw_data = values.tobytes()

collector = runtime.ParallelRegionCollector(capacity=8, hardware_counters=True)
options = runtime.RuntimeSessionOptions(
    parameters=runtime.RuntimeParameters(1), parallel_region_collector=collector
)
context = runtime.RuntimeContext(runtime.KernelContext(runtime.default_opset(18)))
context.set("x", runtime.tensor_from_proto(tensor_proto))
session = runtime.RuntimeSession(model, options)
session.run(context)

report = session.parallel_region_report()
print(f"events={len(report.events)}, dropped={report.dropped_events}")
for event in report.events:
    location = f"{event.file_name}:{event.line}"
    wall_time = "unavailable" if event.wall_time_ns is None else f"{event.wall_time_ns} ns"
    utilization = (
        "unavailable" if event.cpu_utilization is None else f"{event.cpu_utilization:.3f}"
    )
    ipc = "unavailable" if event.ipc is None else f"{event.ipc:.3f}"
    llc = "unavailable" if event.llc_miss_rate is None else f"{event.llc_miss_rate:.3%}"
    counter_time = (
        "unavailable"
        if event.counter_time_enabled is None
        else f"{event.counter_time_running}/{event.counter_time_enabled}"
    )
    print(
        f"{location}: participants requested/admitted/observed="
        f"{event.requested_threads}/{event.admitted_threads}/{event.observed_threads}, "
        f"wall={wall_time}, cpu_utilization={utilization}, "
        f"counters={event.counter_status}, time_running/enabled={counter_time}, "
        f"ipc={ipc}, llc_miss_rate={llc}"
    )
