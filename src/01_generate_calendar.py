"""
BikeFlowBerlin V2 — Step 1: Calendar Data Generation
Generates Berlin public holidays and full calendar data for 2012–2025.
"""

import subprocess, sys

# Auto-install missing libraries
def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

try:
    import pandas as pd
except ImportError:
    install("pandas"); import pandas as pd

try:
    import holidays
except ImportError:
    install("holidays"); import holidays

import os
from datetime import date, timedelta

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(ROOT, "data", "calendar", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

HOLIDAYS_FILE = os.path.join(RAW_DIR, "berlin_holidays_2012_2025.csv")
CALENDAR_FILE = os.path.join(RAW_DIR, "calendar_data_2012_2025.csv")

# ─── 1. Generate Berlin Public Holidays ───────────────────────────────────────
print("\n📅 Generating Berlin public holidays 2012–2025 ...")

berlin_hols = holidays.Germany(prov="BE", years=range(2012, 2026))

holiday_rows = []
for d, name in sorted(berlin_hols.items()):
    holiday_rows.append({"date": pd.Timestamp(d).strftime("%Y-%m-%d"), "holiday_name": name})

df_holidays = pd.DataFrame(holiday_rows)
df_holidays.to_csv(HOLIDAYS_FILE, index=False)
print(f"   Saved {len(df_holidays)} holiday rows → {HOLIDAYS_FILE}")


# ─── 2. Season Helper ─────────────────────────────────────────────────────────
def get_season(d: date) -> str:
    md = (d.month, d.day)
    if (3, 21) <= md <= (6, 20):
        return "Spring"
    elif (6, 21) <= md <= (9, 22):
        return "Summer"
    elif (9, 23) <= md <= (12, 20):
        return "Autumn"
    else:
        return "Winter"


# ─── 3. School Holiday Helper ─────────────────────────────────────────────────
def is_school_holiday(d: date) -> bool:
    m, day = d.month, d.day
    # July & August
    if m in (7, 8):
        return True
    # Winter break: Dec 22 – Jan 5
    if (m == 12 and day >= 22) or (m == 1 and day <= 5):
        return True
    # Easter / Spring break: Mar 10–25
    if m == 3 and 10 <= day <= 25:
        return True
    return False


# ─── 4. Generate Full Calendar ────────────────────────────────────────────────
print("📅 Generating full calendar 2012–2025 ...")

WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Build holiday lookup dict  {date_str: holiday_name}
holiday_lookup = df_holidays.set_index("date")["holiday_name"].to_dict()

start = date(2012, 1, 1)
end   = date(2025, 12, 31)

rows = []
current = start
while current <= end:
    ds = current.strftime("%Y-%m-%d")
    wd = current.weekday()          # 0=Mon … 6=Sun
    hol_name = holiday_lookup.get(ds, None)

    rows.append({
        "date":            ds,
        "year":            current.year,
        "month":           current.month,
        "day":             current.day,
        "weekday":         wd,
        "weekday_name":    WEEKDAY_NAMES[wd],
        "is_weekend":      int(wd >= 5),
        "week_of_year":    current.isocalendar()[1],
        "season":          get_season(current),
        "holiday_name":    hol_name if hol_name else "",
        "is_holiday":      int(hol_name is not None),
        "is_school_holiday": int(is_school_holiday(current)),
    })
    current += timedelta(days=1)

df_cal = pd.DataFrame(rows)
df_cal.to_csv(CALENDAR_FILE, index=False)

# ─── 5. Completion Report ─────────────────────────────────────────────────────
print("\n✅ Step 1 — Calendar COMPLETE")
print(f"📁 Saved: data/calendar/raw/berlin_holidays_2012_2025.csv — {len(df_holidays)} rows")
print(f"📁 Saved: data/calendar/raw/calendar_data_2012_2025.csv — {len(df_cal)} rows")
print(f"📋 Columns: {list(df_cal.columns)}")
print(f"📅 Date range: {df_cal['date'].iloc[0]} to {df_cal['date'].iloc[-1]}")
print("\n⏭️  Ready for Step 2 — Official Counters? Type YES to continue.")
