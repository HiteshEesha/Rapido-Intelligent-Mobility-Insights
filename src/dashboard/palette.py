"""Fixed color assignments shared by every chart in the dashboard.

Values are the validated reference palette (colorblind-safe, checked against
both light and dark chart surfaces) -- colors are assigned by entity and kept
fixed across every chart/filter state, never re-cycled.
"""

from __future__ import annotations

CATEGORICAL = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

SEQUENTIAL_BLUE = [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b",
]

STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

# booking_status is an outcome, not a pure nominal category: Cancelled/Incomplete
# read naturally as "bad", Completed as neutral/good.
BOOKING_STATUS_COLORS = {
    "Completed": CATEGORICAL[0],
    "Cancelled": STATUS["critical"],
    "Incomplete": STATUS["warning"],
}

VEHICLE_TYPE_COLORS = {
    "Bike": CATEGORICAL[2],
    "Auto": CATEGORICAL[3],
    "Cab": CATEGORICAL[4],
}

CITY_COLORS = {
    "Mumbai": CATEGORICAL[0],
    "Delhi": CATEGORICAL[1],
    "Bangalore": CATEGORICAL[2],
    "Chennai": CATEGORICAL[3],
    "Hyderabad": CATEGORICAL[4],
}

# Ordinal severity scales -- one hue family per level, not distinct identities.
TRAFFIC_LEVEL_COLORS = {"Low": STATUS["good"], "Medium": STATUS["warning"], "High": STATUS["critical"]}
WEATHER_COLORS = {"Clear": STATUS["good"], "Rain": STATUS["warning"], "Heavy Rain": STATUS["critical"]}
DEMAND_LEVEL_COLORS = {"Low": STATUS["good"], "Medium": STATUS["warning"], "High": STATUS["critical"]}

DIVERGING = ["#2a78d6", "#f0efec", "#e34948"]
