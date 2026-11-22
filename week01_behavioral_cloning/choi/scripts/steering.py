import math


def camera_deltas(settings):
    """Return signed (left, right) offsets, including for historical run configs."""
    if "left_delta" in settings or "right_delta" in settings:
        if not {"left_delta", "right_delta"} <= settings.keys() or "steering_delta" in settings:
            raise ValueError("Specify both left_delta and right_delta without steering_delta")
        left, right = settings["left_delta"], settings["right_delta"]
    elif "steering_delta" in settings:
        left = settings["steering_delta"]
        right = -left
    else:
        raise ValueError("Missing model_b steering settings: specify left_delta and right_delta")
    if not (math.isfinite(left) and math.isfinite(right) and left >= 0 and right <= 0):
        raise ValueError("Expected finite left_delta >= 0 and right_delta <= 0")
    return left, right
