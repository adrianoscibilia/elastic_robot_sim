# One lifecycle event during a recording run (RR_01 S3.1).
#
# Published on /erd/events by the orchestrator (erd_recording.pipeline) and
# recorded in every bag, so a converter can align segments and a report can
# reconstruct the run timeline without re-parsing controller-state topics.

std_msgs/Header header

# Run identity -----------------------------------------------------------
string run_id                  # <robot>/<YYYYMMDD>/<run_id> folder name
string segment_id              # e.g. "excitation_3", "standstill_1", "identify_sweep_2"

# Event kind ---------------------------------------------------------------
# plan_loaded, cancel, fault, or "<segment.kind>_start"/"<segment.kind>_end"
# for whichever plan segment kind is executing: approach, excitation, return,
# standstill_approach, standstill_return, identify (RR_04 A-3/A-8). The
# standstill hold itself (not a trajectory) uses standstill_start/
# standstill_end instead.
string kind

# Plan identity, when the event concerns a frozen segment (empty otherwise).
string plan_digest

# Free-form detail, e.g. a fault reason or a cancel cause. Empty when unused.
string detail
