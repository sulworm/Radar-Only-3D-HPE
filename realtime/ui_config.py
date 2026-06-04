"""Central UI defaults and display colors."""


DEFAULT_PORT = "COM5"

UI_BASE_FONT_PX = 13
UI_TITLE_FONT_PX = 17
AXIS_LEGEND_TITLE_FONT_PT = 9
AXIS_LEGEND_TEXT_FONT_PT = 8
AXIS_LEGEND_WIDTH = 320
AXIS_LEGEND_HEIGHT = 150

SKELETON_BONES = [
    (0, 7),
    (7, 8),
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (7, 9),
    (9, 10),
    (7, 11),
    (11, 12),
]

# Edit cloud_low/cloud_high for the point-cloud gradient.
# Edit joint/bone for the inferred pose colors.
# Outline colors and sizes keep the 3D overlays legible against each viewport.
DISPLAY_THEMES = {
    "dark": {
        "window_bg": "#111827",
        "panel_bg": "#1f2937",
        "text": "#d1d5db",
        "title": "#f9fafb",
        "muted": "#9ca3af",
        "accent": "#93c5fd",
        "border": "#374151",
        "input_bg": "#111827",
        "input_border": "#4b5563",
        "input_text": "#e5e7eb",
        "button_bg": "#374151",
        "button_hover": "#4b5563",
        "button_text": "#f3f4f6",
        "disabled_text": "#6b7280",
        "disabled_bg": "#1f2937",
        "viewport": "#111827",
        "grid": (107, 114, 128, 120),
        "cloud_low": (0.10, 0.85, 1.00),
        "cloud_high": (1.00, 0.30, 0.55),
        "cloud_outline": (0.02, 0.06, 0.10, 0.42),
        "cloud_size": 6,
        "cloud_outline_size": 8,
        "joint": (1.00, 0.20, 0.18, 1.0),
        "joint_outline": (0.02, 0.06, 0.10, 0.95),
        "joint_size": 7,
        "joint_outline_size": 7,
        "bone": (1.00, 0.75, 0.15, 1.0),
        "bone_outline": (0.02, 0.06, 0.10, 0.92),
        "bone_width": 3,
        "bone_outline_width": 3,
    },
    "light": {
        "window_bg": "#e5e7eb",
        "panel_bg": "#ffffff",
        "text": "#1f2937",
        "title": "#111827",
        "muted": "#6b7280",
        "accent": "#1d4ed8",
        "border": "#d1d5db",
        "input_bg": "#ffffff",
        "input_border": "#9ca3af",
        "input_text": "#111827",
        "button_bg": "#e5e7eb",
        "button_hover": "#d1d5db",
        "button_text": "#1f2937",
        "disabled_text": "#9ca3af",
        "disabled_bg": "#f3f4f6",
        "viewport": "#eef2f7",
        "grid": (71, 85, 105, 110),
        "cloud_low": (0.00, 0.28, 0.60),
        "cloud_high": (0.86, 0.12, 0.25),
        "cloud_outline": (0.04, 0.08, 0.16, 0.52),
        "cloud_size": 6,
        "cloud_outline_size": 8,
        "joint": (0.86, 0.04, 0.12, 1.0),
        "joint_outline": (0.02, 0.08, 0.18, 0.98),
        "joint_size": 7,
        "joint_outline_size": 7,
        "bone": (1.00, 0.55, 0.00, 1.0),
        "bone_outline": (0.02, 0.08, 0.18, 0.96),
        "bone_width": 3,
        "bone_outline_width": 3,
    },
}

STYLE_TEMPLATE = """
QMainWindow, QWidget {
    background: %(window_bg)s;
    color: %(text)s;
    font-size: %(base_font)spx;
}
QWidget#controlPanel {
    background: %(panel_bg)s;
    border-radius: 8px;
}
QLabel#title {
    color: %(title)s;
    font-size: %(title_font)spx;
    font-weight: bold;
}
QLabel#subtitle {
    color: %(muted)s;
    margin-bottom: 4px;
}
QGroupBox {
    border: 1px solid %(border)s;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 8px;
    font-weight: bold;
    color: %(accent)s;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QTextEdit {
    background: %(input_bg)s;
    border: 1px solid %(input_border)s;
    border-radius: 4px;
    padding: %(padding)spx;
    color: %(input_text)s;
}
QPushButton {
    background: %(button_bg)s;
    border: 1px solid %(input_border)s;
    border-radius: 4px;
    padding: %(button_v_padding)spx %(button_h_padding)spx;
    color: %(button_text)s;
}
QPushButton:hover {
    background: %(button_hover)s;
}
QPushButton#startButton {
    background: #047857;
    border-color: #059669;
    font-weight: bold;
}
QPushButton#startButton:hover {
    background: #059669;
}
QPushButton#startButton[collecting="true"] {
    background: #b91c1c;
    border-color: #dc2626;
}
QPushButton#startButton[collecting="true"]:hover {
    background: #dc2626;
}
QPushButton:disabled, QComboBox:disabled, QSpinBox:disabled,
QDoubleSpinBox:disabled, QLineEdit:disabled {
    color: %(disabled_text)s;
    background: %(disabled_bg)s;
}
"""
