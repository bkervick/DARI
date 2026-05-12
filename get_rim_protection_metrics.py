import requests
import pandas as pd
import time
import random

BASE = "https://hoop-explorer.com/api"

YEAR = "2025/26"

# ============================================
# LOAD PLAYER BLOCK DATA (FROM CSV)
# ============================================

player_df = pd.read_csv("players.csv")

blk_lookup = {
    (row["player_name"], row["team"]): row["def_blk"]
    for _, row in player_df.iterrows()
}

# ============================================
# REQUEST SETUP
# ============================================

session = requests.Session()

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}


def safe_get(url, params, retries=5):

    for attempt in range(retries):

        try:
            time.sleep(random.uniform(1.2, 2.4))

            r = session.get(url, params=params, headers=HEADERS, timeout=30)

            if r.status_code == 429:
                raise Exception("Rate limited (429)")

            r.raise_for_status()

            return r.json()

        except Exception as e:
            print(f"Request failed ({attempt+1}/{retries}): {e}")
            time.sleep((attempt + 1) * 5)

    return None


# ============================================
# API CALL
# ============================================

def get_on_off_data(player, team, def_adj_opp, def_fc, maxRank):

    url = f"{BASE}/calculateOnOffStats"

    params = {
        "autoOffQuery": "true",
        "baseQuery": "",
        "gender": "Men",
        "maxRank": maxRank,
        "minRank": 0,
        "onQuery": f'"{player}"',
        "team": team,
        "year": YEAR
    }

    return safe_get(url, params)


# ============================================
# HELPERS
# ============================================

def val(bucket, key):
    try:
        x = bucket[key]["value"]
        return 0 if x is None else x
    except:
        return 0


def safe_div(a, b):
    return a / b if b else 0


def extract_bucket(data, name):
    try:
        return (
            data["responses"][0]
            ["aggregations"]["tri_filter"]
            ["buckets"][name]
        )
    except:
        return {}


# ============================================
# METRICS
# ============================================

def calc_metrics(player, team, def_adj_opp, def_fc, on_bucket, base_bucket):

    # -----------------------
    # ON COURT
    # -----------------------

    on_poss = val(on_bucket, "def_poss")

    on_rim_att = val(on_bucket, "total_def_2prim_attempts")
    on_rim_made = val(on_bucket, "total_def_2prim_made")
    on_2p_att = val(on_bucket, "total_def_2p_attempts")
    on_FG_att = val(on_bucket, "total_def_fga")

    on_rim_fg = safe_div(on_rim_made, on_rim_att)
    on_rim_freq = safe_div(on_rim_att, on_FG_att )
    on_rim_att_allow_per100 = safe_div(on_rim_att, on_poss) * 100

    # PLAYER BLOCK RATE (FROM CSV)
    blk_rate = blk_lookup.get((player, team), 0)
    total_blk = blk_rate * on_2p_att
    rim_blk_rate = safe_div(total_blk, on_rim_att)


    # -----------------------
    # BASELINE (TEAM)
    # -----------------------

    base_poss = val(base_bucket, "def_poss")

    base_rim_att = val(base_bucket, "total_def_2prim_attempts")
    base_rim_made = val(base_bucket, "total_def_2prim_made")
    base_FG_att = val(base_bucket, "total_def_fga")

    # -----------------------
    # OFF ESTIMATE (BASE - ON)
    # -----------------------

    off_poss = base_poss - on_poss

    off_rim_att = base_rim_att - on_rim_att
    off_rim_made = base_rim_made - on_rim_made
    off_FG_att = base_FG_att - on_FG_att

    off_rim_fg = safe_div(off_rim_made, off_rim_att)
    off_rim_freq = safe_div(off_rim_att, off_FG_att)

    # -----------------------
    # DIFFERENTIALS
    # -----------------------

    rim_fg_diff = off_rim_fg - on_rim_fg
    rim_freq_diff = off_rim_freq - on_rim_freq

    # -----------------------
    # COMPOSITE SCORE
    # -----------------------

    rim_protection_score = (
        rim_fg_diff * 100 * 0.40 +
        rim_freq_diff * 100 * 0.40 +
        rim_blk_rate * 100 * 0.20
    )


    return {
        "on_poss": on_poss,
        "on_rim_fg": round(on_rim_fg, 4),
        "on_rim_freq": round(on_rim_freq, 4),
        "on_rim_att_allow_per100": round(on_rim_att_allow_per100, 4),


        "off_poss": off_poss,
        "off_rim_fg": round(off_rim_fg, 4),
        "off_rim_freq": round(off_rim_freq, 4),

        "rim_fg_diff": round(rim_fg_diff, 4),
        "rim_freq_diff": round(rim_freq_diff, 4),
        "rim_blk_rate": round(rim_blk_rate, 4),

        "rim_protection_score": round(rim_protection_score, 2)
    }


# ============================================
# MAIN
# ============================================

results = []

for idx, row in player_df.iterrows():

    player = row["player_name"]
    team = row["team"]
    def_adj_opp = row["def_adj_opp"]
    def_fc = row["def_fc"]

    print(f"[{idx+1}/{len(player_df)}] {player}")
    
    maxRank = 400

    data = get_on_off_data(player, team, def_adj_opp, def_fc, maxRank)
   
    if not data:
        print("  -> failed")
        continue

    on_bucket = extract_bucket(data, "on")
    base_bucket = extract_bucket(data, "baseline")

    # First call — full schedule
    metrics_full = calc_metrics(player, team, def_adj_opp, def_fc, on_bucket, base_bucket)
    # Rename keys
    metrics_full = {f"{k}_full": v for k, v in metrics_full.items()}
   
    print (f"Full schedule stats, done.")

    maxRank = 150

    data2 = get_on_off_data(player, team, def_adj_opp, def_fc, maxRank)
   
    if not data:
        print("  -> failed")
        continue

    on_bucket2 = extract_bucket(data2, "on")
    base_bucket2 = extract_bucket(data2, "baseline")

    metrics_top150 = calc_metrics(player, team, def_adj_opp, def_fc, on_bucket2, base_bucket2)
    metrics_top150 = {f"{k}_top150": v for k, v in metrics_top150.items()}
    print (f"Top 150 stats, done.")

    results.append({
        "player": player,
        "team": team,
        "def_adj_opp": def_adj_opp,
        "def_fc": def_fc,
        **metrics_full,
        **metrics_top150,
    })


# ============================================
# OUTPUT
# ============================================

df = pd.DataFrame(results)

print("\nDONE\n")
print(df.head())

df.to_csv("rim_protection_metrics_with_top150.csv", index=False)

print("\nSaved: rim_protection_metrics_with_top150.csv")

