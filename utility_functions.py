def mood_to_color(mood):
    """
    Converts mood (1 to 100) to a grayscale hex color.
    """
    # Clamp to range
    mood = max(1, min(100, mood))
    # Map mood to grayscale intensity (0 = black, 255 = white)
    intensity = int((mood / 100) * 255)
    hex_value = f'{intensity:02x}'  # 2-digit hex
    return f'#{hex_value}{hex_value}{hex_value}'
