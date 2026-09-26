THEMES = {
    "dark": {
        "BG":       "#0a0a0a",
        "PANEL":    "#181818",
        "GREEN":    "#00e676",
        "CYAN":     "#00d4e8",
        "BLUE":     "#2196f3",
        "WHITE":    "#f5f5f5",
        "GRAY":     "#8a8a8a",
        "GRAY_DIM": "#5a5a5a",
        "DARK":     "#252525",
        "BORDER":   "#333333",
    },
    "light": {
        "BG":       "#f4f4f4",
        "PANEL":    "#ffffff",
        "GREEN":    "#00a854",
        "CYAN":     "#0097a7",
        "BLUE":     "#1565c0",
        "WHITE":    "#1a1a1a",
        "GRAY":     "#5a5a5a",
        "GRAY_DIM": "#9a9a9a",
        "DARK":     "#e8e8e8",
        "BORDER":   "#d0d0d0",
    },
}

_current_theme = "dark"

def get_theme_name():
    return _current_theme

def set_theme(name):
    global _current_theme
    if name in THEMES:
        _current_theme = name

def get_theme():
    return THEMES[_current_theme]