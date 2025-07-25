def print_from_seconds_to_hours(seconds):
    """
    Converts seconds into hours, minutes, and remaining seconds.

    Parameters:
        seconds (int): The total number of seconds to convert.

    Returns:
        str: A string in the format "Xh Ym Zs".
    """
    if seconds < 0:
        raise ValueError("Number of seconds cannot be negative.")

    total_seconds = int(round(seconds))

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    remaining_seconds = total_seconds % 60

    return f"{hours}h {minutes}m {remaining_seconds}s"