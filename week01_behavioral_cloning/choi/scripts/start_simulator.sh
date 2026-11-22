#!/usr/bin/env bash
set -eu

root=$(cd -- "$(dirname -- "$0")/../../.." && pwd)
simulator="$root/beta_simulator_linux/beta_simulator.x86_64"
[[ -x "$simulator" ]] || { echo "Missing executable: $simulator" >&2; exit 1; }
display=:87
if [[ -e /tmp/.X11-unix/X87 || -e /tmp/.X87-lock ]]; then
    echo "Display :87 is already in use. Check the existing simulator window." >&2
    exit 1
fi

session=$(mktemp -d /tmp/udacity-display.XXXXXXXX)
auth="$session/auth"
touch "$auth"
xauth -f "$auth" add "$display" . "$(mcookie)"
echo "Unity log: $session/unity.log"

# The configuration dialog needs more height than the game itself.
Xephyr "$display" -screen 1024x768 -sw-cursor -nocursor -noreset -nolisten tcp -auth "$auth" \
    -title 'Udacity simulator (isolated X display)' >"$session/xephyr.log" 2>&1 &
server_pid=$!
game_pid=
cursor_pid=
cleanup() {
    if [[ -n "$cursor_pid" ]]; then kill "$cursor_pid" 2>/dev/null || true; fi
    if [[ -n "$game_pid" ]]; then kill "$game_pid" 2>/dev/null || true; fi
    kill "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT TERM
export DISPLAY="$display" XAUTHORITY="$auth" LIBGL_ALWAYS_SOFTWARE=1 __GLX_VENDOR_LIBRARY_NAME=mesa
for ((i=0; i<100; i++)); do
    if xdpyinfo >/dev/null 2>&1; then break; fi
    kill -0 "$server_pid" 2>/dev/null || { cat "$session/xephyr.log" >&2; exit 1; }
    sleep 0.1
done
xdpyinfo >/dev/null
"$root/.venv/bin/python" "$(dirname -- "$0")/simulator_cursor.py" &
cursor_pid=$!
"$simulator" -screen-fullscreen 0 -screen-width 800 -screen-height 600 \
    -screen-quality Fastest -logFile "$session/unity.log" &
game_pid=$!
while kill -0 "$game_pid" 2>/dev/null && kill -0 "$server_pid" 2>/dev/null; do
    if xwininfo -name self_driving_car_nanodegree_program 2>/dev/null | grep -q 'Width: 800'; then
        xrandr -s 800x600
        break
    fi
    sleep 0.3
done
wait -n "$server_pid" "$game_pid"
