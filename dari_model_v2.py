###
# DARI — Deterrence-Adjusted Rim Impact
# ======================================
# Full pipeline for building and modeling a college basketball
# rim protection metric using Hoop-Explorer data.
#
# DARI is computed twice — once on full schedule data and once on
# top-150 opponent data only — then combined via:
#
#   1. Empirical Bayes shrinkage on DARI_full: players with no
#      top-150 exposure are shrunk toward the population mean
#      proportionally to their lack of top-150 evidence.
#
#   2. Blend: shrunk DARI_full and DARI_top150 are blended based
#      on the proportion of full-schedule rim attempts that came
#      against top-150 opponents.
#
#   Final DARI = DARI_full_shrunk * (1 - blend_weight)
#              + DARI_top150      * blend_weight
#
# Sections:
#     1. Configuration
#     2. Data Loading
#     3. DARI Construction (single split)
#     4. Blending with Empirical Bayes Shrinkage
#     5. Diagnostics
#     6. Output
###

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from bs4 import BeautifulSoup
from kenpompy.utils import login
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

# --- File path ---
CSV_PATH = r'C:\Users\ra2bk\projects\Hoop-Explorer\rim_protection_metrics_with_top150.csv'

# --- KenPom credentials ---
KP_EMAIL    = 'username'
KP_PASSWORD = 'password'

# --- D1 Baseline Constants ---
# L_full and L_top150 are overridden at runtime with sample-derived medians
L_full   = 0.58  # placeholder
L_top150 = 0.58  # placeholder

# --- Opponent adjustment range ---
# def_adj_opp ranges from 100 (toughest) to 117 (easiest)
OPP_ADJ_FLOOR   = 100
OPP_ADJ_CEILING = 117

# --- Deterrence adjustment stretch ---
# Widens opp_adj_factor from 1.00->1.17 to 1.00->1.35
OPP_ADJ_STRETCH = 0.35

# --- Rim-specific opponent tendency adjustment ---
# Rewards deterrence against more rim-aggressive offenses
# 0.10 = 10% bonus/penalty per std deviation of off_rim_freq
RIM_OPP_WEIGHT = 0.10

# --- Possession value model ---
# Marginal deterrence value = rim PPP - replacement PPP
REPLACEMENT_PPP = 0.60

# --- Foul penalty ---
# def_fc is fouls per 50 possessions -> x2 for per 100
# Marginal cost relative to sample median
FOUL_COST = 0.25

# --- Winsorizing ---
# Applied independently before blending
# Top-150 has a wider distribution due to small samples — use larger threshold
WINSOR_THRESHOLD_FULL   = 15.0
WINSOR_THRESHOLD_TOP150 = 18.0

# --- Empirical Bayes shrinkage ---
# Controls how much top-150 evidence is needed before trusting DARI_full.
# SHRINKAGE_K is expressed in rim attempt units.
# At k top-150 rim attempts: shrinkage = 0.50 (halfway to population mean)
# At 0 top-150 rim attempts: shrinkage = 1.00 (collapses to population mean)
# At 3k top-150 rim attempts: shrinkage = 0.25 (mostly trusts observed)
# Tune between 75 (aggressive) and 200 (conservative).
SHRINKAGE_K = 120

# --- Weighting ---
# Based on full schedule rim attempt count
WEIGHT_CAP = 350

# --- Outlier threshold for diagnostics ---
DARI_OUTLIER_THRESHOLD = 8.0

# --- Column name mapping ---
COL = {
    'player':                   'player',
    'team':                     'team',
    'def_adj_opp':              'def_adj_opp',
    'foul':                     'def_fc',
    'two_pt_dist':              '2P Dist',

    # Full schedule
    'on_rim_fg_full':           'on_rim_fg_full',
    'on_rim_freq_full':         'on_rim_freq_full',
    'off_rim_fg_full':          'off_rim_fg_full',
    'off_rim_freq_full':        'off_rim_freq_full',
    'on_rim_att_full':          'on_rim_att_allow_per100_full',
    'def_possessions_full':     'on_poss_full',
    'block_rate_full':          'rim_blk_rate_full',

    # Top 150 opponents only
    'on_rim_fg_top150':         'on_rim_fg_top150',
    'on_rim_freq_top150':       'on_rim_freq_top150',
    'off_rim_fg_top150':        'off_rim_fg_top150',
    'off_rim_freq_top150':      'off_rim_freq_top150',
    'on_rim_att_top150':        'on_rim_att_allow_per100_top150',
    'def_possessions_top150':   'on_poss_top150',
    'block_rate_top150':        'rim_blk_rate_top150',
}

# --- Team name mapping: Hoop-Explorer -> KenPom ---
NAME_MAP = {
    'A&M-Corpus Christi':   'Texas A&M Corpus Chris',
    'Alcorn':               'Alcorn St.',
    'App State':            'Appalachian St.',
    'Ark.-Pine Bluff':      'Arkansas Pine Bluff',
    'Army West Point':      'Army',
    'Bethune-Cookman':      'Bethune Cookman',
    'Boston U.':            'Boston University',
    'CSU Bakersfield':      'Cal St. Bakersfield',
    'California Baptist':   'Cal Baptist',
    'Central Ark.':         'Central Arkansas',
    'Central Conn. St.':    'Central Connecticut',
    'Central Mich.':        'Central Michigan',
    'Charleston So.':       'Charleston Southern',
    'Col. of Charleston':   'Charleston',
    'ETSU':                 'East Tennessee St.',
    'Eastern Ill.':         'Eastern Illinois',
    'Eastern Ky.':          'Eastern Kentucky',
    'Eastern Mich.':        'Eastern Michigan',
    'Eastern Wash.':        'Eastern Washington',
    'FDU':                  'Fairleigh Dickinson',
    'FGCU':                 'Florida Gulf Coast',
    'Fla. Atlantic':        'Florida Atlantic',
    'Ga. Southern':         'Georgia Southern',
    'Gardner-Webb':         'Gardner Webb',
    'Grambling':            'Grambling St.',
    'LMU (CA)':             'Loyola Marymount',
    'Lamar University':     'Lamar',
    'Loyola Maryland':      'Loyola MD',
    'Miami (FL)':           'Miami FL',
    'Miami (OH)':           'Miami OH',
    'Middle Tenn.':         'Middle Tennessee',
    'Mississippi Val.':     'Mississippi Valley St.',
    'N.C. A&T':             'North Carolina A&T',
    'N.C. Central':         'North Carolina Central',
    'NC State':             'N.C. State',
    'NIU':                  'Northern Illinois',
    'North Ala.':           'North Alabama',
    'Northern Ariz.':       'Northern Arizona',
    'Northern Colo.':       'Northern Colorado',
    'Northern Ky.':         'Northern Kentucky',
    'Ole Miss':             'Mississippi',
    'Omaha':                'Nebraska Omaha',
    'Prairie View':         'Prairie View A&M',
    'Queens (NC)':          'Queens',
    'SFA':                  'Stephen F. Austin',
    "Saint Mary's (CA)":    "Saint Mary's",
    'Sam Houston':          'Sam Houston St.',
    'Seattle U':            'Seattle',
    'South Fla.':           'South Florida',
    'Southeast Mo. St.':    'Southeast Missouri',
    'Southeastern La.':     'Southeastern Louisiana',
    'Southern California':  'USC',
    'Southern Ill.':        'Southern Illinois',
    'Southern Ind.':        'Southern Indiana',
    'Southern Miss.':       'Southern Miss',
    'Southern U.':          'Southern',
    "St. John's (NY)":      "St. John's",
    'St. Thomas (MN)':      'St. Thomas',
    'UAlbany':              'Albany',
    'UConn':                'Connecticut',
    'UIC':                  'Illinois Chicago',
    'UIW':                  'Incarnate Word',
    'ULM':                  'Louisiana Monroe',
    'UMES':                 'Maryland Eastern Shore',
    'UNCW':                 'UNC Wilmington',
    'UNI':                  'Northern Iowa',
    'UT Martin':            'Tennessee Martin',
    'UTRGV':                'UT Rio Grande Valley',
    'West Ga.':             'West Georgia',
    'Western Caro.':        'Western Carolina',
    'Western Ill.':         'Western Illinois',
    'Western Ky.':          'Western Kentucky',
    'Western Mich.':        'Western Michigan',
}


# =============================================================================
# 2. DATA LOADING
# =============================================================================

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} player records.")
    print(f"Columns: {list(df.columns)}\n")
    return df


def scrape_kenpom_2pt_dist(email: str, password: str) -> pd.DataFrame:
    """
    Scrapes the KenPom offensive teamstats page and returns a dataframe
    with Team and 2P Dist columns.
    """
    print("Logging in to KenPom...")
    browser  = login(email, password)
    url      = 'https://kenpom.com/teamstats.php'
    response = browser.get(url)
    soup     = BeautifulSoup(response.text, 'html.parser')
    table    = soup.find('table', {'id': 'ratings-table'})

    rows = []
    for tr in table.find_all('tr'):
        cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
        if cells:
            rows.append(cells)

    df_kp            = pd.DataFrame(rows)
    df_kp.columns    = df_kp.iloc[0]
    df_kp            = df_kp.drop(0).reset_index(drop=True)
    df_kp            = df_kp[['Team', '2P Dist']].copy()
    df_kp['Team']    = df_kp['Team'].str.replace(r'\d+', '', regex=True).str.strip()
    df_kp['2P Dist'] = pd.to_numeric(df_kp['2P Dist'], errors='coerce')
    df_kp            = df_kp.dropna()

    print(f"KenPom: scraped {len(df_kp)} teams.\n")
    return df_kp


def merge_kenpom(df: pd.DataFrame, df_kp: pd.DataFrame) -> pd.DataFrame:
    """
    Applies team name mapping and merges KenPom 2P Dist into player dataframe.
    """
    df['team_kp'] = df[COL['team']].replace(NAME_MAP)
    df_merged     = df.merge(df_kp, left_on='team_kp', right_on='Team', how='left')

    missing = df_merged[df_merged['2P Dist'].isna()][COL['team']].unique()
    if len(missing) > 0:
        print(f"WARNING: {len(missing)} unmatched teams — 2P Dist will be NaN:")
        for t in missing:
            print(f"  '{t}'")
    else:
        print(f"KenPom join: all {len(df_merged)} players matched successfully.")

    return df_merged


# =============================================================================
# 3. DARI CONSTRUCTION (SINGLE SPLIT)
# =============================================================================

def build_dari_split(
    df: pd.DataFrame,
    split: str,
    L: float,
) -> pd.Series:
    """
    Computes raw (pre-winsorized) DARI for a single data split.

    Parameters:
        df    — player dataframe with split-specific columns
        split — 'full' or 'top150'
        L     — sample median rim FG% for this split

    Returns a Series of raw DARI values and stores intermediate
    columns with _{split} suffix for diagnostics.
    """

    rim_ppp                   = L * 2
    marginal_deterrence_value = rim_ppp - REPLACEMENT_PPP

    on_rim_fg    = df[COL[f'on_rim_fg_{split}']]
    on_rim_freq  = df[COL[f'on_rim_freq_{split}']]
    off_rim_freq = df[COL[f'off_rim_freq_{split}']]
    on_rim_att   = df[COL[f'on_rim_att_{split}']]

    # --- Schedule strength adjustment ---
    raw_factor     = (OPP_ADJ_CEILING + OPP_ADJ_FLOOR - df[COL['def_adj_opp']]) / 100
    raw_range      = (OPP_ADJ_CEILING - OPP_ADJ_FLOOR) / 100
    opp_adj_factor = 1 + (raw_factor - 1) * (OPP_ADJ_STRETCH / raw_range)

    # --- Rim-specific opponent tendency adjustment ---
    off_freq_median = off_rim_freq.median()
    off_freq_std    = off_rim_freq.std()
    rim_opp_factor  = 1 + (
        (off_rim_freq - off_freq_median) / off_freq_std
    ) * RIM_OPP_WEIGHT

    # --- Accuracy suppression ---
    opp_adj_offset      = OPP_ADJ_CEILING - df[COL['def_adj_opp']]
    X_sc                = opp_adj_offset.values.reshape(-1, 1)
    y_sc                = on_rim_fg.values
    sc_reg              = LinearRegression().fit(X_sc, y_sc)
    scaling_constant    = sc_reg.coef_[0]
    expected_rim_fg_pct = L + (opp_adj_offset * scaling_constant)
    A_adj               = expected_rim_fg_pct - on_rim_fg

    # --- Deterrence ---
    D_adj = (
        (off_rim_freq - on_rim_freq) * 100
    ) * opp_adj_factor * rim_opp_factor

    # --- DARI components ---
    deterrence_pts     = D_adj * marginal_deterrence_value
    accuracy_pts       = on_rim_att * A_adj * 2
    foul_per100        = df[COL['foul']] * 2
    median_foul_per100 = foul_per100.median()
    foul_penalty       = (foul_per100 - median_foul_per100) * FOUL_COST

    dari = deterrence_pts + accuracy_pts - foul_penalty

    # --- Store intermediate columns for diagnostics ---
    df[f'D_adj_{split}']                   = D_adj.round(4)
    df[f'A_adj_{split}']                   = A_adj.round(4)
    df[f'opp_adj_factor_{split}']          = opp_adj_factor.round(4)
    df[f'rim_opp_factor_{split}']          = rim_opp_factor.round(4)
    df[f'expected_rim_fg_pct_{split}']     = expected_rim_fg_pct.round(4)
    df[f'deterrence_contribution_{split}'] = deterrence_pts.round(4)
    df[f'accuracy_contribution_{split}']   = accuracy_pts.round(4)
    df[f'foul_contribution_{split}']       = (-foul_penalty).round(4)
    df[f'scaling_constant_{split}']        = scaling_constant

    print(f"  [{split}] scaling constant: {scaling_constant:.4f}  "
          f"marginal det. value: {marginal_deterrence_value:.3f}")

    return dari


# =============================================================================
# 4. BLENDING WITH EMPIRICAL BAYES SHRINKAGE
# =============================================================================

def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds DARI_full and DARI_top150, applies empirical Bayes shrinkage
    to DARI_full, then blends with DARI_top150.

    Empirical Bayes shrinkage:
        shrinkage_factor = SHRINKAGE_K / (SHRINKAGE_K + top150_rim_att)

        DARI_full_shrunk = (1 - shrinkage_factor) * DARI_full
                         + shrinkage_factor        * population_mean

        At 0 top-150 rim attempts:   shrinkage = 1.0 -> collapses to mean
        At SHRINKAGE_K attempts:     shrinkage = 0.5 -> halfway to mean
        At 3*SHRINKAGE_K attempts:   shrinkage = 0.25 -> mostly trusts observed

    Blend:
        blend_weight = top150_rim_att / full_rim_att (clipped to [0, 1])

        DARI = DARI_full_shrunk * (1 - blend_weight)
             + DARI_top150      * blend_weight
    """

    print("Building DARI (full schedule)...")
    dari_full_raw   = build_dari_split(df, 'full', L_full)
    df['DARI_full'] = dari_full_raw.clip(
        lower=-WINSOR_THRESHOLD_FULL,
        upper=WINSOR_THRESHOLD_FULL
    )
    clipped_full = (dari_full_raw.abs() > WINSOR_THRESHOLD_FULL).sum()
    print(f"  [{clipped_full} players clipped at ±{WINSOR_THRESHOLD_FULL}]")

    print("\nBuilding DARI (top-150 opponents)...")
    dari_top150_raw   = build_dari_split(df, 'top150', L_top150)
    df['DARI_top150'] = dari_top150_raw.clip(
        lower=-WINSOR_THRESHOLD_TOP150,
        upper=WINSOR_THRESHOLD_TOP150
    )
    clipped_top150 = (dari_top150_raw.abs() > WINSOR_THRESHOLD_TOP150).sum()
    print(f"  [{clipped_top150} players clipped at ±{WINSOR_THRESHOLD_TOP150}]")

    # --- Rim attempt counts ---
    df['total_rim_att_full'] = (
        df[COL['on_rim_att_full']] / 100
    ) * df[COL['def_possessions_full']]

    df['total_rim_att_top150'] = (
        df[COL['on_rim_att_top150']] / 100
    ) * df[COL['def_possessions_top150']]

    # --- Empirical Bayes shrinkage on DARI_full ---
    # Shrinks players with little top-150 evidence toward the population mean
    population_mean_full   = df['DARI_full'].mean()
    df['shrinkage_factor'] = SHRINKAGE_K / (
        SHRINKAGE_K + df['total_rim_att_top150']
    )
    df['DARI_full_shrunk'] = (
        (1 - df['shrinkage_factor']) * df['DARI_full'] +
        df['shrinkage_factor']       * population_mean_full
    )

    # --- Blend weight ---
    df['blend_weight'] = (
        df['total_rim_att_top150'] /
        (df['total_rim_att_full'] + 1e-6)
    ).clip(upper=1.0)

    # --- Consistency signal ---
    df['consistency'] = df['DARI_top150'] - df['DARI_full']

    # --- Final blended DARI ---
    df['DARI'] = (
        df['DARI_full_shrunk'] * (1 - df['blend_weight']) +
        df['DARI_top150']      * df['blend_weight']
    )

    # --- Sample weight (based on full schedule rim attempts) ---
    df['weight'] = df['total_rim_att_full'].clip(upper=WEIGHT_CAP) / WEIGHT_CAP

    print(f"\nPopulation mean DARI_full: {population_mean_full:.3f}")
    print(f"\nShrinkage factor distribution:")
    print(df['shrinkage_factor'].describe().round(3))
    print(f"\nBlend weight distribution:")
    print(df['blend_weight'].describe().round(3))
    print(f"\nConsistency distribution:")
    print(df['consistency'].describe().round(3))

    return df


# =============================================================================
# 5. DIAGNOSTICS
# =============================================================================

def derive_baselines(df: pd.DataFrame) -> None:
    """
    Derives L_full and L_top150 from sample medians and prints
    verification diagnostics.
    """

    global L_full, L_top150

    L_full   = df[COL['on_rim_fg_full']].median()
    L_top150 = df[COL['on_rim_fg_top150']].median()

    print(f"Sample-derived L (full):   {L_full:.4f}")
    print(f"Sample-derived L (top150): {L_top150:.4f}")

    r, p = pearsonr(df[COL['def_adj_opp']], df[COL['on_rim_fg_full']])
    print(f"Correlation def_adj_opp vs on_rim_fg_full: {r:.4f} (p={p:.4f})")

    d_check = (df[COL['off_rim_freq_full']] - df[COL['on_rim_freq_full']]) * 100
    r2, p2  = pearsonr(df[COL['off_rim_freq_full']], d_check)
    print(f"Correlation off_rim_freq vs D_adj (full):  {r2:.4f} (p={p2:.4f})")

    print(f"\nFoul variable (def_fc) preview:")
    print(df[COL['foul']].describe().round(4))
    print()


def run_diagnostics(df: pd.DataFrame) -> None:

    print("=" * 60)
    print("DARI DISTRIBUTION DIAGNOSTICS")
    print("=" * 60)

    for label, col in [
        ('Full schedule (raw)',    'DARI_full'),
        ('Full schedule (shrunk)', 'DARI_full_shrunk'),
        ('Top-150 only',           'DARI_top150'),
        ('Final blended',          'DARI'),
    ]:
        print(f"\n--- {label} ---")
        print(df[col].describe().round(3))
        print(f"  Skewness: {df[col].skew():.3f}   "
              f"Kurtosis: {df[col].kurt():.3f}")

    print("\n--- Shrinkage Diagnostics ---")
    print(f"  SHRINKAGE_K:           {SHRINKAGE_K} rim attempts")
    print(f"  shrinkage_factor mean: {df['shrinkage_factor'].mean():.3f}")
    print(f"  shrinkage_factor std:  {df['shrinkage_factor'].std():.3f}")
    zero_top150 = (df['total_rim_att_top150'] == 0).sum()
    print(f"  Players with 0 top-150 rim attempts: {zero_top150}")
    low_top150  = (df['total_rim_att_top150'] < SHRINKAGE_K).sum()
    print(f"  Players below shrinkage threshold "
          f"(<{SHRINKAGE_K} top-150 att): {low_top150} / {len(df)}")

    print("\n--- Shrinkage Effect: Biggest Movers ---")
    df['shrinkage_delta'] = df['DARI_full_shrunk'] - df['DARI_full']
    print("  Most pulled down (high DARI_full, low top-150 sample):")
    print(df.nsmallest(5, 'shrinkage_delta')[[
        COL['player'], COL['team'], 'DARI_full',
        'DARI_full_shrunk', 'shrinkage_factor', 'total_rim_att_top150'
    ]].round(3).to_string(index=False))

    print("\n--- Blend Weight Distribution ---")
    print(df['blend_weight'].describe().round(3))

    print("\n--- Consistency (DARI_top150 - DARI_full) ---")
    print(df['consistency'].describe().round(3))

    print("\n--- Top 10 Most Consistent ---")
    print(df.nlargest(10, 'consistency')[[
        COL['player'], COL['team'], 'DARI_full',
        'DARI_top150', 'consistency', 'blend_weight', 'shrinkage_factor'
    ]].round(3).to_string(index=False))

    print("\n--- Bottom 10 Most Inconsistent ---")
    print(df.nsmallest(10, 'consistency')[[
        COL['player'], COL['team'], 'DARI_full',
        'DARI_top150', 'consistency', 'blend_weight', 'shrinkage_factor'
    ]].round(3).to_string(index=False))

    print("\n--- Component Contributions — Full Schedule (mean) ---")
    print(f"  Deterrence:           "
          f"{df['deterrence_contribution_full'].mean():.3f} pts/100")
    print(f"  Accuracy suppression: "
          f"{df['accuracy_contribution_full'].mean():.3f} pts/100")
    print(f"  Foul penalty:         "
          f"{df['foul_contribution_full'].mean():.3f} pts/100")

    print("\n--- Component Contributions — Top-150 (mean) ---")
    print(f"  Deterrence:           "
          f"{df['deterrence_contribution_top150'].mean():.3f} pts/100")
    print(f"  Accuracy suppression: "
          f"{df['accuracy_contribution_top150'].mean():.3f} pts/100")
    print(f"  Foul penalty:         "
          f"{df['foul_contribution_top150'].mean():.3f} pts/100")

    print(f"\n--- Sample Weight Distribution ---")
    full_weight = (df['weight'] >= 1.0).sum()
    print(f"  Players at full weight (>={WEIGHT_CAP} rim attempts): "
          f"{full_weight} / {len(df)}")
    print(f"  Mean weight: {df['weight'].mean():.3f}")

    print(f"\n--- Outliers in Final DARI (|DARI| > {DARI_OUTLIER_THRESHOLD}) ---")
    outliers = df[df['DARI'].abs() > DARI_OUTLIER_THRESHOLD].copy()
    if len(outliers) == 0:
        print("  None found.")
    else:
        print(f"  {len(outliers)} outlier(s):")
        print(
            outliers[[
                COL['player'], COL['team'],
                'total_rim_att_full', 'weight', 'DARI'
            ]]
            .sort_values('DARI', ascending=False)
            .round(3)
            .to_string(index=False)
        )

    print("\n--- Sanity Check: Top 10 Final DARI ---")
    print(df.nlargest(10, 'DARI')[[
        COL['player'], COL['team'], 'DARI', 'DARI_full',
        'DARI_full_shrunk', 'DARI_top150',
        'shrinkage_factor', 'blend_weight', 'weight'
    ]].round(3).to_string(index=False))

    print("\n--- Sanity Check: Bottom 10 Final DARI ---")
    print(df.nsmallest(10, 'DARI')[[
        COL['player'], COL['team'], 'DARI', 'DARI_full',
        'DARI_full_shrunk', 'DARI_top150',
        'shrinkage_factor', 'blend_weight', 'weight'
    ]].round(3).to_string(index=False))

    print("=" * 60 + "\n")


# =============================================================================
# 6. OUTPUT
# =============================================================================

def build_output(df: pd.DataFrame) -> pd.DataFrame:
    """
    Saves full results to CSV and prints ranked tables.
    DARI is the primary deliverable.
    weight >= 0.85 is the recommended filter for published rankings.
    """

    output_cols = [
        COL['player'],
        COL['team'],
        'DARI',
        'DARI_full',
        'DARI_full_shrunk',
        'DARI_top150',
        'consistency',
        'shrinkage_factor',
        'blend_weight',
        'deterrence_contribution_full',
        'accuracy_contribution_full',
        'foul_contribution_full',
        'deterrence_contribution_top150',
        'accuracy_contribution_top150',
        'foul_contribution_top150',
        'D_adj_full',
        'A_adj_full',
        'D_adj_top150',
        'A_adj_top150',
        'opp_adj_factor_full',
        'rim_opp_factor_full',
        'total_rim_att_full',
        'total_rim_att_top150',
        'weight',
        COL['two_pt_dist'],
    ]

    output_cols = [c for c in output_cols if c in df.columns]
    results     = df[output_cols].sort_values('DARI', ascending=False).round(4)

    out_path = 'dari_results.csv'
    results.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")

    display_cols = [
        COL['player'], COL['team'], 'DARI', 'DARI_full',
        'DARI_full_shrunk', 'DARI_top150',
        'consistency', 'shrinkage_factor', 'blend_weight', 'weight'
    ]
    display_cols = [c for c in display_cols if c in results.columns]

    print("\n--- Top 20 DARI (all players) ---")
    print(results[display_cols].head(20).to_string(index=False))

    print("\n--- Top 20 DARI (weight >= 0.85) ---")
    ranked = results[results['weight'] >= 0.85].copy()
    print(ranked[display_cols].head(20).to_string(index=False))

    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    # 1. Scrape KenPom 2P Dist
    df_kp = scrape_kenpom_2pt_dist(KP_EMAIL, KP_PASSWORD)

    # 2. Load player data
    df = load_data(CSV_PATH)

    # 3. Merge KenPom 2P Dist by team
    df = merge_kenpom(df, df_kp)

    # 4. Derive baselines (sets L_full and L_top150 globally)
    derive_baselines(df)

    # 5. Build blended DARI with empirical Bayes shrinkage
    df = build_target(df)

    # 6. Diagnostics
    run_diagnostics(df)

    # 7. Output
    results = build_output(df)
