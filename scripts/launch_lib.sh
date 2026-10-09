# Shared machinery of the unattended launchers (sourced, not run):
# scripts/round6_run.sh and scripts/round6_retrain.sh (FR_03 T-5).
#
#   interpreters   pick_py, SIM_PY / NN_PY / HELP
#   identity       tree_digest, git_heads (the PREFLIGHT_OK ticket's key)
#   stages         say, stage_timeout, stage: .done/.failed resumability,
#                  STATUS.md, timing.tsv, per-stage timeouts, the deadline
#                  and the 10 GB hard stop
#   doctor         doctor_machine: disk, sleep inhibition, setsid, flock, lock
#   ticket         write_ticket, find_ticket
#   launch         supervise, launch_detached (setsid, systemd-inhibit, ionice)
#
# The caller sets SIM_DIR, NN_DIR, LOCK and, before `stage`, RUN, LOGS,
# STATUS, TIMEOUTS, STARTED, DEADLINE_HOURS, MODE and HARD_STOP=0.
# STAGE_SIGNAL (default INT) is what `timeout` sends a stage that runs over.

pick_py() {
  local d="$1"
  [ -x "$d/.venv/bin/python" ] && { echo "$d/.venv/bin/python"; return; }
  if command -v uv >/dev/null 2>&1 && [ -f "$d/pyproject.toml" ]; then echo "uv run --project $d python"; return; fi
  command -v python3 >/dev/null 2>&1 && { echo python3; return; }
  echo python
}
read -r -a SIM_PY <<< "${SIM_PY_CMD:-$(pick_py "$SIM_DIR")}"
read -r -a NN_PY  <<< "${NN_PY_CMD:-$(pick_py "$NN_DIR")}"
HELP=("${SIM_PY[@]}" "$SIM_DIR/scripts/launch_helpers.py")

tree_digest() {
  local d
  for d in "$SIM_DIR" "$NN_DIR"; do
    git -C "$d" rev-parse HEAD
    git -C "$d" diff HEAD --binary
    git -C "$d" ls-files -o --exclude-standard -z -- '*.py' '*.sh' '*.yaml' '*.yml' '*.toml' \
      | (cd "$d" && xargs -0 -r sha256sum)
  done 2>/dev/null | sha256sum | cut -c1-16
}
git_heads() { echo "sim=$(git -C "$SIM_DIR" rev-parse --short HEAD) nn=$(git -C "$NN_DIR" rev-parse --short HEAD)"; }

# ------------------------------------------------------------- the stages ---
say() { printf '%s | %s\n' "$(date '+%m-%d %H:%M:%S')" "$*" | tee -a "$STATUS"; }

stage_timeout() {
  local t="" base="${1%-e[0-9]*}"
  [ -f "$TIMEOUTS" ] && t=$(awk -F'\t' -v s="$1" -v b="$base" '$1==s||$1==b{print $2; exit}' "$TIMEOUTS")
  if [ -n "$t" ]; then echo "${t}s"; elif [ "$MODE" = preflight ]; then echo "${PREFLIGHT_STAGE_TIMEOUT:-60m}"; else echo "${DEFAULT_STAGE_TIMEOUT:-4h}"; fi
}

# stage NAME LOGFILE CMD...: run once (skipped when NAME.done exists), timed, logged.
stage() {
  local name="$1" log="$2"; shift 2
  [ -f "$RUN/$name.done" ] && { say "skip  $name"; return 0; }
  local used=$(( ($(date +%s) - STARTED) / 3600 ))
  if [ "$used" -ge "$DEADLINE_HOURS" ]; then say "SKIP  $name (past ${DEADLINE_HOURS}h)"; touch "$RUN/$name.failed"; return 1; fi
  local free; free=$(df -BG --output=avail "$SIM_DIR" | tail -1 | tr -dc '0-9')
  if [ "${free:-0}" -lt 10 ]; then say "HARD STOP before $name -- only ${free}G free (< 10 GB)"; HARD_STOP=1; return 1; fi
  local limit; limit=$(stage_timeout "$name")
  say "START $name (timeout $limit)"
  local t0; t0=$(date +%s)
  rm -f "$RUN/$name.failed"
  if timeout --signal="${STAGE_SIGNAL:-INT}" --kill-after=180 "$limit" "$@" >>"$LOGS/$log" 2>&1; then
    printf '%s\t%s\n' "$name" "$(( $(date +%s) - t0 ))" >> "$RUN/timing.tsv"
    touch "$RUN/$name.done"; say "OK    $name ($(( $(date +%s) - t0 ))s)"; return 0
  fi
  local code=$?
  say "FAIL  $name (exit $code) -- logs/$log"; touch "$RUN/$name.failed"; return 1
}

# ---------------------------------------------------------------- doctor ----
# Machine checks; the caller's doctor defines ok/bad/warn and the rc they set.
doctor_machine() {
  local context="$1" min_free="$2"
  local free; free=$(df -BG --output=avail "$SIM_DIR" | tail -1 | tr -dc '0-9')
  [ "${free:-0}" -ge "$min_free" ] && ok "${free}G free (need ${min_free})" || bad "only ${free}G free, need ${min_free}"
  local masked=1 t
  for t in sleep.target suspend.target hibernate.target hybrid-sleep.target; do
    [ "$(systemctl is-enabled "$t" 2>/dev/null)" = masked ] || masked=0
  done
  if command -v systemd-inhibit >/dev/null && systemd-inhibit --what=sleep:idle --who=doctor --why=check true 2>/dev/null; then
    ok "systemd-inhibit can block sleep"
  elif [ $masked -eq 1 ]; then
    ok "sleep/suspend/hibernate targets are masked"
  elif [ "$context" = launch ]; then
    bad "cannot inhibit sleep (systemd-inhibit refused, sleep targets not masked): the machine may suspend mid-run"
  else
    warn "cannot inhibit sleep from this session (launch will FAIL on this unless run where polkit allows it, or the sleep targets are masked)"
  fi
  command -v gsettings >/dev/null && echo "  info  GNOME idle sleep on AC: $(gsettings get org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 2>/dev/null)"
  command -v setsid >/dev/null && ok "setsid available (survives logout)" || bad "setsid missing"
  command -v flock >/dev/null && ok "flock available (one run at a time)" || bad "flock missing"
  if [ -z "${HOLDING_LOCK:-}" ] && [ -e "$LOCK" ] && ! flock -n "$LOCK" true; then bad "another run holds $LOCK"; fi
}

# ----------------------------------------------------------- launch ticket ---
# write_ticket FILE [EXTRA_LINE...]: the green preflight's key for this tree.
write_ticket() {
  local out="$1"; shift
  {
    echo "heads=$(git_heads | tr ' ' ',')"
    echo "digest=$(tree_digest)"
    local line; for line in "$@"; do echo "$line"; done
    echo "finished=$(date -Is)"
  } > "$out"
}

# find_ticket [REQUIRED_LINE...]: newest runs/*/PREFLIGHT_OK for the current
# heads and tree digest that also carries every REQUIRED_LINE; prints its run dir.
find_ticket() {
  local heads digest t line ok
  heads="heads=$(git_heads | tr ' ' ',')"; digest="digest=$(tree_digest)"
  for t in $(ls -1t "$SIM_DIR"/runs/*/PREFLIGHT_OK 2>/dev/null); do
    grep -qx "$heads" "$t" && grep -qx "$digest" "$t" || continue
    ok=1
    for line in "$@"; do grep -qx -- "$line" "$t" || { ok=0; break; }; done
    [ $ok -eq 1 ] && { dirname "$t"; return 0; }
  done
  return 1
}

# ------------------------------------------------------------- supervisor ---
# supervise MODE: run `work MODE` up to six times; exit 2 from work is a hard stop.
supervise() {
  local mode="$1" n=0 rc
  while [ $n -lt 6 ]; do
    n=$((n+1))
    echo "--- supervisor attempt $n $(date)"
    work "$mode"; rc=$?
    [ $rc -eq 0 ] && break
    [ $rc -eq 2 ] && { echo "hard stop, not retrying"; break; }
    sleep 60
  done
}

# launch_detached SCRIPT WHO WHY: `bash SCRIPT run` detached from the terminal,
# sleep-inhibited, OOM-protected; prints the "launched RUN_ID=... / watch:" lines.
launch_detached() {
  local script="$1" who="$2" why="$3"
  local boot="$SIM_DIR/runs/$RUN_ID.boot.log"
  local inhibit=(systemd-inhibit --what=sleep:idle:shutdown:handle-lid-switch
                 --who="$who" --why="$why" --mode=block)
  setsid nohup ionice -c2 -n0 "${inhibit[@]}" bash -c \
    'echo -500 > /proc/self/oom_score_adj 2>/dev/null; exec bash "$0" run' \
    "$script" \
    </dev/null >>"$boot" 2>&1 &
  disown
  echo "launched RUN_ID=$RUN_ID"
  echo "watch:  tail -f $SIM_DIR/runs/$RUN_ID/STATUS.md"
  echo "boot :  $boot"
}
