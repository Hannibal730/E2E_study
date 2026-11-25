import argparse
import asyncio
from contextlib import ExitStack, suppress
import os
import secrets
import signal
import subprocess
import tempfile
import base64
from io import BytesIO
from pathlib import Path

from aiohttp import web
from Xlib import X, display
from Xlib.error import ConnectionClosedError
from Xlib.ext import shape


def driving_app(checkpoint_path, target_speed):
    from PIL import Image
    import socketio
    import torch

    from train import image_transform, make_model, select_device

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    device = select_device("auto")
    model = make_model(config["model"], weights=None)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    transform = image_transform(config["preprocessing"])
    print(f"Loaded {checkpoint_path} ({device}); target speed {target_speed}", flush=True)

    server = socketio.AsyncServer(async_mode="aiohttp", async_handlers=False)
    app = web.Application()
    server.attach(app)

    async def steer(sid, steering, throttle):
        await server.emit("steer", {"steering_angle": str(steering), "throttle": str(throttle)}, room=sid)

    @server.on("connect")
    async def connect(sid, environ):
        print("Simulator connected", flush=True)
        await steer(sid, 0.0, 0.0)

    @server.on("telemetry")
    async def telemetry(sid, data):
        if not data:
            await server.emit("manual", {}, room=sid)
            return
        with Image.open(BytesIO(base64.b64decode(data["image"]))) as frame:
            tensor = transform(frame.convert("RGB")).unsqueeze(0).to(device)
        with torch.inference_mode():
            steering = model(tensor).clamp(-1, 1).item()
        speed = float(data["speed"])
        throttle = max(-1.0, min(1.0, 0.12 * (target_speed - speed)))
        await steer(sid, steering, throttle)

    return app


def create_cursor(connection):
    screen = connection.screen()
    root = screen.root
    width, height = 22, 28
    arrow = [(1, 1), (1, 23), (7, 17), (12, 26), (16, 24), (11, 15), (20, 15)]
    mask = root.create_pixmap(width, height, 1)
    gc = mask.create_gc(foreground=0)
    mask.fill_rectangle(gc, 0, 0, width, height)
    gc.change(foreground=1)
    mask.fill_poly(gc, X.Complex, X.CoordModeOrigin, arrow)

    image = root.create_pixmap(width, height, screen.root_depth)
    ink = image.create_gc(foreground=screen.white_pixel)
    image.fill_rectangle(ink, 0, 0, width, height)
    ink.change(foreground=screen.black_pixel, line_width=2)
    image.poly_line(ink, X.CoordModeOrigin, arrow + arrow[:1])
    window = root.create_window(0, 0, width, height, 0, X.CopyFromParent,
                                background_pixmap=image, override_redirect=True)
    window.set_wm_name("Udacity cursor")
    window.shape_mask(shape.SO.Set, shape.SK.Bounding, 0, 0, mask)
    window.shape_rectangles(shape.SO.Set, shape.SK.Input, X.Unsorted, 0, 0, [])
    window.map()
    return window


def stop_process(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


async def launch_window():
    binary = Path(__file__).resolve().parents[1] / "beta_simulator_linux" / "beta_simulator.x86_64"
    if not os.access(binary, os.X_OK):
        raise SystemExit(f"Missing executable: {binary}")
    if Path("/tmp/.X11-unix/X87").exists() or Path("/tmp/.X87-lock").exists():
        raise SystemExit("Display :87 is in use. Check the existing simulator window.")

    session = Path(tempfile.mkdtemp(prefix="udacity-display."))
    auth = session / "auth"
    auth.touch(mode=0o600)
    subprocess.run(["xauth", "-f", auth, "add", ":87", ".", secrets.token_hex(16)], check=True)
    print(f"Unity log: {session / 'unity.log'}", flush=True)
    with ExitStack() as stack:
        log = stack.enter_context((session / "xephyr.log").open("w"))
        xephyr = subprocess.Popen(
            ["Xephyr", ":87", "-screen", "1024x768", "-sw-cursor", "-nocursor", "-noreset",
             "-nolisten", "tcp", "-auth", auth, "-title", "Udacity simulator (isolated X display)"],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        stack.callback(stop_process, xephyr)
        # Xephyr inherited the host display. Unity and the cursor use only its private display.
        os.environ.update(DISPLAY=":87", XAUTHORITY=str(auth), LIBGL_ALWAYS_SOFTWARE="1",
                          __GLX_VENDOR_LIBRARY_NAME="mesa")
        for _ in range(100):
            if xephyr.poll() is not None:
                raise SystemExit(f"Xephyr exited. See {log.name}")
            if subprocess.run(["xdpyinfo"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
                break
            await asyncio.sleep(0.1)
        else:
            raise SystemExit(f"Display :87 did not become ready. See {log.name}")

        conn = display.Display()
        stack.enter_context(suppress(ConnectionClosedError))
        stack.callback(conn.close)
        root = conn.screen().root
        cursor = create_cursor(conn)
        game = subprocess.Popen([binary, "-screen-fullscreen", "0", "-screen-width", "800",
                                 "-screen-height", "600", "-screen-quality", "Fastest",
                                 "-logFile", session / "unity.log"], start_new_session=True)
        stack.callback(stop_process, game)
        next_resize_check = 0.0
        loop = asyncio.get_running_loop()
        while game.poll() is None and xephyr.poll() is None:
            pointer = root.query_pointer()
            cursor.configure(x=pointer.root_x - 1, y=pointer.root_y - 1, stack_mode=X.Above)
            conn.flush()
            if loop.time() >= next_resize_check:
                info = subprocess.run(["xwininfo", "-name", "self_driving_car_nanodegree_program"],
                                      capture_output=True, text=True)
                if info.returncode == 0 and "Width: 800" in info.stdout:
                    subprocess.run(["xrandr", "-s", "800x600"], check=True)
                    next_resize_check = float("inf")
                else:
                    next_resize_check = loop.time() + 0.3
            await asyncio.sleep(1 / 60)
        if game.returncode:
            raise SystemExit(f"Unity exited with status {game.returncode}. See {session / 'unity.log'}")


async def run(args):
    runner = None
    if args.checkpoint:
        runner = web.AppRunner(driving_app(args.checkpoint, args.target_speed))
        await runner.setup()
    try:
        if runner is not None:
            await web.TCPSite(runner, "127.0.0.1", 4567).start()
            print("Driving server ready on 127.0.0.1:4567", flush=True)
        if args.server_only:
            await asyncio.Event().wait()
        else:
            await launch_window()
    finally:
        if runner is not None:
            await runner.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the Udacity simulator and optional driving server.")
    parser.add_argument("checkpoint", type=Path, nargs="?", help="PyTorch best.pt for autonomous driving")
    parser.add_argument("--target-speed", type=float, default=8.0)
    parser.add_argument("--server-only", action="store_true", help="Serve a simulator in another window")
    args = parser.parse_args()
    if args.server_only and args.checkpoint is None:
        parser.error("--server-only requires a checkpoint")
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass
