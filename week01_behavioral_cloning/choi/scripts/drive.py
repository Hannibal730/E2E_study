import argparse
import base64
from io import BytesIO
from pathlib import Path

from aiohttp import web
from PIL import Image
import socketio
import torch

from dataset import image_transform
from train import make_model, select_device


def driving_app(checkpoint_path, target_speed):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drive the Udacity Term 1 simulator with a BC checkpoint.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--target-speed", type=float, default=8.0)
    args = parser.parse_args()
    web.run_app(driving_app(args.checkpoint, args.target_speed), host="127.0.0.1", port=4567)
