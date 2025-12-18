def build_plant_state(sensor_out: dict, vision_out: dict) -> dict:
    """
    Łączy wyniki sensor AI + vision AI w jeden stan.
    """

    alerts = []

    if sensor_out.get("too_bright_now"):
        alerts.append("TOO_BRIGHT_NOW")

    if sensor_out.get("forgot_to_water"):
        alerts.append("FORGOT_TO_WATER")

    if sensor_out.get("too_dark_today"):
        alerts.append("TOO_DARK_TODAY")

    if sensor_out.get("worth_relocating"):
        alerts.append("WORTH_RELOCATING")

    health = vision_out.get("top_bucket", "unknown")

    if alerts:
        overall = "ALERT"
    elif health.startswith("unhealthy"):
        overall = "WARNING"
    else:
        overall = "OK"

    return {
        "overall_state": overall,
        "health_from_image": health,
        "alerts": alerts,
    }
