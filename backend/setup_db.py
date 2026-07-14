import sqlite3
from datetime import datetime, timedelta
import random

conn = sqlite3.connect("railway_mock.db")
cursor = conn.cursor()

# ── Table 1: track_segments ──────────────────────────────────────
cursor.execute("""
CREATE TABLE IF NOT EXISTS track_segments (
    segment_id      TEXT PRIMARY KEY,
    track_id        TEXT,
    mile_post_start REAL,
    mile_post_end   REAL,
    bfi_value       REAL,
    ballast_volume  REAL,
    status          TEXT,
    survey_date     TEXT,
    subdivision     TEXT
)
""")

subdivisions = ["Plainview", "Amarillo", "Clovis", "Belen", "Needles"]
segments = []
for i in range(1, 51):
    seg_id = f"SEG-{i:04d}"
    track = f"TRK-{random.choice(['A','B','C'])}{random.randint(1,5)}"
    mile_start = round(random.uniform(0, 500), 2)
    bfi = round(random.uniform(0.3, 1.2), 3)
    volume = round(random.uniform(200, 800), 2)
    status = "CRITICAL" if bfi > 1.0 else "WARNING" if bfi > 0.8 else "OK"
    date = (datetime.now() - timedelta(days=random.randint(0, 90))).strftime("%Y-%m-%d")
    sub = random.choice(subdivisions)
    segments.append((seg_id, track, mile_start, round(mile_start+0.5, 2),
                     bfi, volume, status, date, sub))

cursor.executemany("""
INSERT OR REPLACE INTO track_segments VALUES (?,?,?,?,?,?,?,?,?)
""", segments)

# ── Table 2: track_assets ────────────────────────────────────────
cursor.execute("""
CREATE TABLE IF NOT EXISTS track_assets (
    asset_id        TEXT PRIMARY KEY,
    asset_type      TEXT,
    asset_name      TEXT,
    segment_id      TEXT,
    mile_post       REAL,
    subdivision     TEXT,
    inspection_date TEXT,
    condition       TEXT
)
""")

asset_names = {
    "BRIDGE": ["Red River Bridge", "Canadian River Bridge", "Pecos Bridge",
               "Rio Grande Bridge", "Cimarron Bridge"],
    "TUNNEL": ["Raton Tunnel", "Abo Canyon Tunnel", "Apache Canyon Tunnel"]
}

assets = []
for i, (atype, names) in enumerate(asset_names.items()):
    for j, name in enumerate(names):
        asset_id = f"ASSET-{i:02d}{j:02d}"
        seg_id = f"SEG-{random.randint(1,50):04d}"
        mile = round(random.uniform(0, 500), 2)
        sub = random.choice(subdivisions)
        insp_date = (datetime.now() - timedelta(
            days=random.randint(0,180))).strftime("%Y-%m-%d")
        condition = random.choice(["GOOD", "FAIR", "POOR"])
        assets.append((asset_id, atype, name, seg_id,
                       mile, sub, insp_date, condition))

cursor.executemany("""
INSERT OR REPLACE INTO track_assets VALUES (?,?,?,?,?,?,?,?)
""", assets)

# ── Table 3: geometry_surveys ────────────────────────────────────
cursor.execute("""
CREATE TABLE IF NOT EXISTS geometry_surveys (
    survey_id       TEXT PRIMARY KEY,
    segment_id      TEXT,
    survey_date     TEXT,
    centerline_x    REAL,
    centerline_y    REAL,
    top_of_rail_l   REAL,
    top_of_rail_r   REAL,
    gauge_mm        REAL,
    cross_level_mm  REAL
)
""")

surveys = []
for i in range(1, 81):
    survey_id = f"SURV-{i:04d}"
    seg_id = f"SEG-{random.randint(1,50):04d}"
    date = (datetime.now() - timedelta(
        days=random.randint(0,60))).strftime("%Y-%m-%d")
    cx = round(random.uniform(-105, -97), 6)
    cy = round(random.uniform(33, 38), 6)
    tor_l = round(random.uniform(1.0, 3.5), 3)
    tor_r = round(random.uniform(1.0, 3.5), 3)
    gauge = round(random.uniform(1430, 1440), 1)
    cross = round(random.uniform(-5, 5), 2)
    surveys.append((survey_id, seg_id, date, cx, cy,
                    tor_l, tor_r, gauge, cross))

cursor.executemany("""
INSERT OR REPLACE INTO geometry_surveys VALUES (?,?,?,?,?,?,?,?,?)
""", surveys)

# ── Table 4: maintenance_log ─────────────────────────────────────
cursor.execute("""
CREATE TABLE IF NOT EXISTS maintenance_log (
    log_id      TEXT PRIMARY KEY,
    segment_id  TEXT,
    work_type   TEXT,
    crew_size   INTEGER,
    cost_usd    REAL,
    work_date   TEXT,
    completed   INTEGER
)
""")

work_types = ["TAMPING", "BALLAST_CLEANING", "UNDERCUTTING",
              "SPOT_REPAIR", "FULL_REPLACEMENT"]

logs = []
for i in range(1, 61):
    log_id = f"LOG-{i:04d}"
    seg_id = f"SEG-{random.randint(1,50):04d}"
    wtype = random.choice(work_types)
    crew = random.randint(3, 12)
    cost = round(random.uniform(5000, 120000), 2)
    date = (datetime.now() - timedelta(
        days=random.randint(0,120))).strftime("%Y-%m-%d")
    completed = random.choice([0, 1])
    logs.append((log_id, seg_id, wtype, crew, cost, date, completed))

cursor.executemany("""
INSERT OR REPLACE INTO maintenance_log VALUES (?,?,?,?,?,?,?)
""", logs)

conn.commit()
conn.close()
print("✅ Railway mock database created: railway_mock.db")
print("Tables: track_segments, track_assets, geometry_surveys, maintenance_log")