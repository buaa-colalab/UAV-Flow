"""UAV instruction formatting shared by dataset loading and HTTP inference."""


def format_uav_instruction(proprio, instruction):
    """Preserve the trained state text: x/y/z and yaw in degrees, one decimal."""
    state = ",".join(str(round(value, 1)) for value in proprio)
    return f"Current State: {state}, What action should the uav take to {instruction}?"
