# geo.py — points, radius, bbox, nearest-k with real coordinates.
#
# Four German cities stored with their real lat/lon (the [lat, lon]
# array encoding; a {"lat": …, "lon": …} map encodes the same point).
# Distances are haversine kilometres:
#
#   radius 600 km from central Berlin (52.52, 13.40):
#     berlin 0.000000, potsdam 26.621424, hamburg 255.120591,
#     munchen 503.833264 — nearest first, inclusive boundary.
#   bbox (47..55, 5..15): all four, key order, the 0.0 sentinel
#     (a box has no center to measure from).
#   nearest 2: berlin, potsdam — exact haversine order.
#
# These are the same points and tolerances the engine's golden geo
# fixture asserts (~1e-6 km).
#
# Run: python examples/geo.py   (after `maturin develop`)

# docs:begin:geo
from corvid import Db

CITIES = [
    ("berlin", 52.52, 13.40),
    ("potsdam", 52.40, 13.06),
    ("hamburg", 53.55, 9.99),
    ("munchen", 48.14, 11.58),
]

with Db.open_memory() as db:
    places = db.collection("places")
    for name, lat, lon in CITIES:
        places.insert(name, {"name": name, "loc": [lat, lon]})
    places.create_geo_index("loc")

    def show(label, hits):
        inside = " ".join(f"{h.key} {h.distance_km:.6f}km" for h in hits)
        print(f"{label:<34} [{inside}]")

    show("within 600km of Berlin:",
         places.geo_within_radius("loc", 52.52, 13.40, 600.0))
    show("bbox 47..55N, 5..15E:",
         places.geo_within_bbox("loc", 47, 5, 55, 15))
    show("nearest 2 to Berlin:",
         places.geo_nearest("loc", 52.52, 13.40, 2))

    places.close()
# docs:end:geo
