import os
import warnings
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import joblib
from joblib import parallel_backend   # <-- add this

# (and you can remove)
# from sklearn.experimental import enable_hist_gradient_boosting


from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score

warnings.filterwarnings("ignore")

# ===============================
# CONFIG
# ===============================
TRAIN_SEASONS = list(range(2010, 2023))  # 2010–2022
VALID_SEASON = 2023                      # validate on 2023
CURRENT_SEASON = 2025
ROLL_N = 5

MODEL_PATH = "nfl_model_v2.joblib"      # bumped to avoid old cache collisions

SPORTS_BLAZE_KEY = os.getenv("SPORTS_BLAZE_KEY", "sb1ry0ir9esayt9yl0len0o")

# ===============================
# TEAM NORMALIZATION
# ===============================
TEAM_MAP = {
    # AFC East
    "bills": "BUF", "buffalo bills": "BUF", "buf": "BUF",
    "dolphins": "MIA", "miami dolphins": "MIA", "mia": "MIA",
    "patriots": "NE", "new england patriots": "NE", "ne": "NE", "nwe": "NE",
    "jets": "NYJ", "new york jets": "NYJ", "nyj": "NYJ",
    # AFC North
    "ravens": "BAL", "baltimore ravens": "BAL", "bal": "BAL",
    "bengals": "CIN", "cincinnati bengals": "CIN", "cin": "CIN",
    "browns": "CLE", "cleveland browns": "CLE", "cle": "CLE",
    "steelers": "PIT", "pittsburgh steelers": "PIT", "pit": "PIT",
    # AFC South
    "texans": "HOU", "houston texans": "HOU", "hou": "HOU",
    "colts": "IND", "indianapolis colts": "IND", "ind": "IND",
    "jaguars": "JAX", "jacksonville jaguars": "JAX", "jax": "JAX", "jac": "JAX",
    "titans": "TEN", "tennessee titans": "TEN", "ten": "TEN",
    # AFC West
    "broncos": "DEN", "denver broncos": "DEN", "den": "DEN",
    "chiefs": "KC", "kansas city chiefs": "KC", "kc": "KC", "kan": "KC",
    "raiders": "LV", "las vegas raiders": "LV", "lv": "LV", "oak": "LV", "oakland raiders": "LV",
    "chargers": "LAC", "los angeles chargers": "LAC", "lac": "LAC", "sd": "LAC", "san diego chargers": "LAC",
    # NFC East
    "cowboys": "DAL", "dallas cowboys": "DAL", "dal": "DAL",
    "giants": "NYG", "new york giants": "NYG", "nyg": "NYG",
    "eagles": "PHI", "philadelphia eagles": "PHI", "phi": "PHI",
    "commanders": "WAS", "washington commanders": "WAS", "washington": "WAS", "was": "WAS", "wft": "WAS",
    # NFC North
    "bears": "CHI", "chicago bears": "CHI", "chi": "CHI",
    "lions": "DET", "detroit lions": "DET", "det": "DET",
    "packers": "GB", "green bay packers": "GB", "gb": "GB", "gnb": "GB",
    "vikings": "MIN", "minnesota vikings": "MIN", "min": "MIN",
    # NFC South
    "falcons": "ATL", "atlanta falcons": "ATL", "atl": "ATL",
    "panthers": "CAR", "carolina panthers": "CAR", "car": "CAR",
    "saints": "NO", "new orleans saints": "NO", "no": "NO", "nor": "NO",
    "buccaneers": "TB", "tampa bay buccaneers": "TB", "tb": "TB", "tam": "TB",
    # NFC West
    "cardinals": "ARI", "arizona cardinals": "ARI", "ari": "ARI",
    "rams": "LA", "los angeles rams": "LA", "la": "LA", "stl": "LA", "st. louis rams": "LA",
    "49ers": "SF", "san francisco 49ers": "SF", "sf": "SF", "sfo": "SF",
    "seahawks": "SEA", "seattle seahawks": "SEA", "sea": "SEA",
}

def normalize_team(name: str) -> str:
    if not name:
        raise ValueError("Empty team name.")
    key = name.strip().lower().replace(".", "").replace("-", " ").replace("the ", "")
    if key in TEAM_MAP:
        return TEAM_MAP[key]
    return name.upper()  # allow "KC", "SF", etc.

# ===============================
# DATA LOADING
# ===============================
def load_schedules(train_seasons, valid_season, current_season):
    import nfl_data_py as nfl
    def prep(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["game_date"] = pd.to_datetime(df.get("gameday"), errors="coerce")
        if "home_score" in df.columns and "away_score" in df.columns:
            df = df[df["home_score"].notna() & df["away_score"].notna()]
        return df
    train_df = prep(nfl.import_schedules(train_seasons))
    valid_df = prep(nfl.import_schedules([valid_season]))
    cur_df   = prep(nfl.import_schedules([current_season]))
    return train_df, valid_df, cur_df

# ===============================
# TEAM-LONG FEATURES
# ===============================
def to_team_long(df_sched: pd.DataFrame) -> pd.DataFrame:
    home = df_sched.rename(columns={
        "home_team": "team", "away_team": "opponent",
        "home_score": "points_for", "away_score": "points_against"
    }).copy()
    home["is_home"] = 1
    away = df_sched.rename(columns={
        "away_team": "team", "home_team": "opponent",
        "away_score": "points_for", "home_score": "points_against"
    }).copy()
    away["is_home"] = 0
    cols = ["season","week","game_id","game_date","team","opponent",
            "points_for","points_against","is_home"]
    df = pd.concat([home[cols], away[cols]], ignore_index=True)
    df["point_diff"] = df["points_for"] - df["points_against"]
    df["win"] = (df["point_diff"] > 0).astype(int)
    df = df.sort_values(["team","game_date","week"]).reset_index(drop=True)
    return df

def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['team_win_pct'] = (df.groupby('team')['win']
                          .expanding().mean()
                          .reset_index(level=0, drop=True)
                          .shift(1).fillna(0.5))
    wr_lookup = (df[['team','game_date','team_win_pct']]
                 .rename(columns={'team':'opponent','team_win_pct':'opp_prior_wr'})
                 .sort_values(['opponent','game_date'])
                 .drop_duplicates(['opponent','game_date']))
    df = df.sort_values(['opponent','game_date'])
    df = df.merge(wr_lookup, on=['opponent','game_date'], how='left')
    df['opponent_strength'] = (df.groupby('team')['opp_prior_wr']
                               .apply(lambda s: s.ffill().fillna(0.5)).values)
    home_perf = df[df['is_home']==1].groupby('team')['win'].mean()
    away_perf = df[df['is_home']==0].groupby('team')['win'].mean()
    df['home_away_diff'] = df['team'].map((home_perf - away_perf).fillna(0))
    return df

def add_rolling_features(df: pd.DataFrame, n: int = ROLL_N) -> pd.DataFrame:
    df = df.sort_values(["team","game_date","week"]).copy()
    def roll_mean(s): return s.rolling(n, min_periods=1).mean().shift(1)
    g = df.groupby("team", group_keys=False)
    df["form_point_diff"]     = g["point_diff"].apply(roll_mean).reset_index(level=0, drop=True)
    df["form_win_rate"]       = g["win"].apply(roll_mean).reset_index(level=0, drop=True)
    df["form_points_for"]     = g["points_for"].apply(roll_mean).reset_index(level=0, drop=True)
    df["form_points_against"] = g["points_against"].apply(roll_mean).reset_index(level=0, drop=True)
    df["rest_days"] = (
        g["game_date"].apply(lambda s: s.diff().shift(1).dt.days)
         .reset_index(level=0, drop=True).fillna(7)
    )
    return df

# ===============================
# ELO
# ===============================
def build_elo_features(df_sched: pd.DataFrame, k=20, hfa=55) -> pd.DataFrame:
    teams = pd.unique(df_sched[['home_team','away_team']].values.ravel('K'))
    elo: Dict[str, float] = {t: 1500.0 for t in teams}
    rows = []
    df = df_sched.sort_values('game_date')
    for _, g in df.iterrows():
        ht, at = g['home_team'], g['away_team']
        eh, ea = elo[ht], elo[at]
        rows.append({'game_id': g['game_id'], 'team': ht, 'elo_pre': eh})
        rows.append({'game_id': g['game_id'], 'team': at, 'elo_pre': ea})
        if pd.notna(g.get('home_score')) and pd.notna(g.get('away_score')):
            Rh = 10 ** ((eh + hfa - ea) / 400)
            ph = Rh / (Rh + 1)
            sh = 1.0 if g['home_score'] > g['away_score'] else 0.0
            elo[ht] = eh + k * (sh - ph)
            elo[at] = ea + k * ((1.0 - sh) - (1 - ph))
    return pd.DataFrame(rows)

# ===============================
# GAME-LEVEL FEATURES (defensive)
# ===============================
def make_game_features(df_sched: pd.DataFrame, team_long: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    X = df_sched.copy()

    # lines
    X["spread_line"] = pd.to_numeric(X.get("spread_line"), errors="coerce")
    X["total_line"]  = pd.to_numeric(
        X.get("over_under_line") if "over_under_line" in X.columns else X.get("total_line"),
        errors="coerce"
    )

    # environment
    for c in ['temp','wind']:
        X[c] = pd.to_numeric(X.get(c), errors='coerce')
    roof = X.get('roof').fillna('outdoors').str.lower()
    surface = X.get('surface').fillna('unknown').str.lower()
    X['is_indoor']  = roof.isin(['dome','closed','indoor']).astype(int)
    X['is_grass']   = surface.str.contains('grass').astype(int)
    X['wind_hi']    = (X['wind'] >= 15).astype(int)
    X['temp_cold']  = (X['temp'] <= 32).astype(int)
    X['temp_hot']   = (X['temp'] >= 85).astype(int)

    # per-side features (incl. elo_pre if present)
    home_feats = team_long.rename(columns={
        "team":"home_team","form_point_diff":"home_form_pd","form_win_rate":"home_wr",
        "form_points_for":"home_pf","form_points_against":"home_pa","rest_days":"home_rest",
        "opponent_strength":"home_sos","home_away_diff":"home_ha_diff","elo_pre":"home_elo_pre"
    })[["game_id","home_team","home_form_pd","home_wr","home_pf","home_pa","home_rest",
        "home_sos","home_ha_diff","home_elo_pre"]].copy()
    away_feats = team_long.rename(columns={
        "team":"away_team","form_point_diff":"away_form_pd","form_win_rate":"away_wr",
        "form_points_for":"away_pf","form_points_against":"away_pa","rest_days":"away_rest",
        "opponent_strength":"away_sos","home_away_diff":"away_ha_diff","elo_pre":"away_elo_pre"
    })[["game_id","away_team","away_form_pd","away_wr","away_pf","away_pa","away_rest",
        "away_sos","away_ha_diff","away_elo_pre"]].copy()

    X = X.merge(home_feats, on=["game_id","home_team"], how="left")
    X = X.merge(away_feats, on=["game_id","away_team"], how="left")

    # ensure columns exist (prevents KeyError like 'home_rest')
    expected = [
        "home_form_pd","home_wr","home_pf","home_pa","home_rest","home_sos","home_ha_diff","home_elo_pre",
        "away_form_pd","away_wr","away_pf","away_pa","away_rest","away_sos","away_ha_diff","away_elo_pre"
    ]
    for c in expected:
        if c not in X.columns:
            X[c] = np.nan

    # diffs
    X["pd_diff"]      = X["home_form_pd"] - X["away_form_pd"]
    X["wr_diff"]      = X["home_wr"] - X["away_wr"]
    X["pf_diff"]      = X["home_pf"] - X["away_pf"]
    X["pa_diff"]      = X["home_pa"] - X["away_pa"]
    X["rest_diff"]    = X["home_rest"] - X["away_rest"]
    X["sos_diff"]     = X["home_sos"] - X["away_sos"]
    X["ha_diff_diff"] = X["home_ha_diff"] - X["away_ha_diff"]
    X["elo_diff"]     = X["home_elo_pre"] - X["away_elo_pre"]

    # situational flags
    X['short_week']      = ((X['home_rest'] < 5) | (X['away_rest'] < 5)).astype(int)
    X['division_game']   = (X['home_team'].str[:1] == X['away_team'].str[:1]).astype(int)  # crude proxy
    X['denver_altitude'] = ((X['home_team'] == 'DEN') | (X['away_team'] == 'DEN')).astype(int)
    X['elo_spread_interact'] = X['elo_diff'] * X['spread_line']

    y = (X["home_score"] > X["away_score"]).astype(int) if "home_score" in X.columns else None
    return X, (y.values if y is not None else None)

# ===============================
# MODEL SAVE / LOAD (schema-aware)
# ===============================
def save_model(model, feature_cols, path=MODEL_PATH):
    payload = {"model": model, "feature_cols": list(feature_cols)}
    joblib.dump(payload, path)
    print(f"Model saved to {path} (features: {len(feature_cols)})")

def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        return None
    payload = joblib.load(path)
    if isinstance(payload, dict) and "model" in payload:
        return payload
    # backward compat if only model was saved
    return {"model": payload, "feature_cols": getattr(payload, "feature_names_in_", None)}

# ===============================
# TRAIN (time-aware + calibration)
# ===============================
from sklearn.calibration import CalibratedClassifierCV

def train_calibrated_hgb(X_train: pd.DataFrame, y_train: np.ndarray, dates: Optional[pd.Series] = None):
    # keep temporal order
    if dates is not None:
        order = np.argsort(pd.to_datetime(dates).values)
        X_train = X_train.iloc[order]
        y_train = y_train[order]

    tscv = TimeSeriesSplit(n_splits=5)

    base = HistGradientBoostingClassifier(
        learning_rate=0.07,
        max_iter=600,
        min_samples_leaf=20,
        random_state=42
    )

    param_dist = {
        'learning_rate': [0.03, 0.05, 0.07, 0.1],
        'max_iter': [400, 600, 800],
        'max_depth': [None, 6, 8, 12],
        'l2_regularization': [0.0, 0.1, 0.5, 1.0],
        'min_samples_leaf': [10, 20, 30, 40],
    }

    rs = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=20,
        scoring='roc_auc',
        cv=tscv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    # use threads to avoid loky pickling issues
    with parallel_backend("threading"):
        rs.fit(X_train, y_train)

    print(f"RandomizedSearch best AUC (time-CV): {rs.best_score_:.3f}")
    best = rs.best_estimator_

    # --- version-safe calibrator ---
    try:
        calib = CalibratedClassifierCV(estimator=best, method='isotonic', cv=tscv)  # sklearn ≥1.6
    except TypeError:
        calib = CalibratedClassifierCV(base_estimator=best, method='isotonic', cv=tscv)  # sklearn ≤1.5

    calib.fit(X_train, y_train)
    return calib

    # use threads to avoid pickling (fixes BrokenProcessPool)
    rs = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=20,
        scoring='roc_auc',
        cv=tscv,
        random_state=42,
        n_jobs=-1,       # keep parallelism
        verbose=1
    )

    # Threading backend sidesteps the numpy._core import issue
    with parallel_backend("threading"):
        rs.fit(X_train, y_train)

    print(f"RandomizedSearch best AUC (time-CV): {rs.best_score_:.3f}")

    best = rs.best_estimator_

    # Calibrator (keep single-process to be extra safe across sklearn versions)
    calib = CalibratedClassifierCV(base_estimator=best, method='isotonic', cv=tscv)
    calib.fit(X_train, y_train)

    return calib

# ===============================
# SPORTS BLAZE (nested shape)
# ===============================
def _sportsblaze_daily_schedule(date_str: str):
    if not SPORTS_BLAZE_KEY:
        return []
    url = f"https://api.sportsblaze.com/nfl/v1/schedule/daily/{date_str}.json?key={SPORTS_BLAZE_KEY}"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print("SportsBlaze API error:", e)
        return []
    games = []
    for g in data.get("games", []):
        home_name = (g.get("teams", {}).get("home", {}) or {}).get("name", "")
        away_name = (g.get("teams", {}).get("away", {}) or {}).get("name", "")
        home_abbr = (g.get("teams", {}).get("home", {}) or {}).get("abbr")
        away_abbr = (g.get("teams", {}).get("away", {}) or {}).get("abbr")
        home = (home_abbr or home_name or "").strip()
        away = (away_abbr or away_name or "").strip()
        if home and away:
            games.append((home, away))
    return games

# ===============================
# PIPELINE: LOAD → FEATURE → TRAIN/LOAD → EVAL
# ===============================
print("Loading NFL data...")
train_sched, valid_sched, cur_sched = load_schedules(TRAIN_SEASONS, VALID_SEASON, CURRENT_SEASON)

print("Creating features...")

# Elo
train_elo = build_elo_features(train_sched)
valid_elo = build_elo_features(valid_sched)

# Team-long bases
train_long_base = add_rolling_features(add_advanced_features(to_team_long(train_sched)), n=ROLL_N)
valid_long_base = add_rolling_features(add_advanced_features(to_team_long(valid_sched)), n=ROLL_N)

# Merge Elo
train_long = train_long_base.merge(train_elo, on=['game_id','team'], how='left')
valid_long = valid_long_base.merge(valid_elo, on=['game_id','team'], how='left')

# Game-level features
X_train_full, y_train = make_game_features(train_sched, train_long)
X_valid_full, y_valid = make_game_features(valid_sched, valid_long)

# Feature list (order matters)
FEATURE_COLS = [
    "pd_diff","wr_diff","pf_diff","pa_diff","rest_diff","sos_diff","ha_diff_diff",
    "spread_line","total_line",
    "is_indoor","is_grass","wind_hi","temp_cold","temp_hot",
    "short_week","division_game","denver_altitude",
    "elo_diff","elo_spread_interact",
]

# Ensure presence + numeric
for df in (X_train_full, X_valid_full):
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = np.nan
    for c in FEATURE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Neutral fills
neutral_fill = {
    "pd_diff":0.0,"wr_diff":0.0,"pf_diff":0.0,"pa_diff":0.0,"rest_diff":0.0,"sos_diff":0.0,"ha_diff_diff":0.0,
    "spread_line":0.0,"total_line":45.0,
    "is_indoor":0.0,"is_grass":0.0,"wind_hi":0.0,"temp_cold":0.0,"temp_hot":0.0,
    "short_week":0.0,"division_game":0.0,"denver_altitude":0.0,
    "elo_diff":0.0,"elo_spread_interact":0.0,
}
X_train = X_train_full[FEATURE_COLS].fillna(value=neutral_fill).reset_index(drop=True)
X_valid = X_valid_full[FEATURE_COLS].fillna(value=neutral_fill).reset_index(drop=True)

# Align targets
mask_train = pd.notna(y_train)
X_train = X_train.loc[mask_train].reset_index(drop=True)
y_train = y_train[mask_train]
mask_valid = pd.notna(y_valid)
X_valid = X_valid.loc[mask_valid].reset_index(drop=True)
y_valid = y_valid[mask_valid]

# Train or load (schema-checked)
loaded = load_model()
needs_retrain = True
pipeline = None
if loaded is not None:
    old_cols = loaded.get("feature_cols")
    if old_cols is not None and set(old_cols) == set(FEATURE_COLS):
        pipeline = loaded["model"]
        needs_retrain = False
        print("Loading saved model... (feature schema matched)")
    else:
        print("Saved model feature schema differs; retraining...")

if needs_retrain:
    print("Training tuned HGB + isotonic calibration...")
    # dates aligned to training rows used
    train_dates = train_sched.loc[train_sched["home_score"].notna() & train_sched["away_score"].notna(), "gameday"]
    pipeline = train_calibrated_hgb(X_train, y_train, dates=train_dates)
    save_model(pipeline, FEATURE_COLS)

# Evaluate
valid_pred = pipeline.predict_proba(X_valid[FEATURE_COLS])[:, 1]
valid_binary = (valid_pred >= 0.5).astype(int)
print(f"Validation AUC on {VALID_SEASON}: {roc_auc_score(y_valid, valid_pred):.3f}")
print(f"Validation Accuracy: {accuracy_score(y_valid, valid_binary):.3f}")

# ===============================
# SNAPSHOT FOR MANUAL MATCHUPS
# ===============================
all_sched_for_snapshot = pd.concat([train_sched, valid_sched, cur_sched], ignore_index=True)
all_elo = build_elo_features(all_sched_for_snapshot)
latest_long = add_rolling_features(add_advanced_features(to_team_long(all_sched_for_snapshot)), n=ROLL_N)
latest_long = latest_long.merge(all_elo, on=['game_id','team'], how='left')
LATEST = latest_long.sort_values(["team","game_date","week"]).groupby("team").tail(1)

def _row_from_teams(h_abbr: str, a_abbr: str) -> pd.DataFrame:
    def get(team):
        row = LATEST[LATEST["team"] == team]
        if row.empty:
            return dict(
                form_point_diff=0.0, form_win_rate=0.5, form_points_for=21.0,
                form_points_against=21.0, rest_days=7.0, opponent_strength=0.5,
                home_away_diff=0.0, elo_pre=1500.0
            )
        r = row.iloc[-1]
        def safe(v, d): 
            return float(v) if pd.notna(v) else d
        return dict(
            form_point_diff=safe(r.get("form_point_diff"), 0.0),
            form_win_rate=safe(r.get("form_win_rate"), 0.5),
            form_points_for=safe(r.get("form_points_for"), 21.0),
            form_points_against=safe(r.get("form_points_against"), 21.0),
            rest_days=safe(r.get("rest_days"), 7.0),
            opponent_strength=safe(r.get("opponent_strength"), 0.5),
            home_away_diff=safe(r.get("home_away_diff"), 0.0),
            elo_pre=safe(r.get("elo_pre"), 1500.0)
        )
    H, A = get(h_abbr), get(a_abbr)
    row = pd.DataFrame([{
        "pd_diff": H["form_point_diff"] - A["form_point_diff"],
        "wr_diff": H["form_win_rate"] - A["form_win_rate"],
        "pf_diff": H["form_points_for"] - A["form_points_for"],
        "pa_diff": H["form_points_against"] - A["form_points_against"],
        "rest_diff": H["rest_days"] - A["rest_days"],
        "sos_diff": H["opponent_strength"] - A["opponent_strength"],
        "ha_diff_diff": H["home_away_diff"] - A["home_away_diff"],
        "spread_line": 0.0, "total_line": 45.0,
        "is_indoor": 0.0, "is_grass": 0.0, "wind_hi": 0.0, "temp_cold": 0.0, "temp_hot": 0.0,
        "short_week": 0.0, "division_game": 0.0, "denver_altitude": 1.0 if (h_abbr=="DEN" or a_abbr=="DEN") else 0.0,
        "elo_diff": H["elo_pre"] - A["elo_pre"],
        "elo_spread_interact": 0.0
    }])
    # ensure all feature columns and order
    for c in FEATURE_COLS:
        if c not in row.columns:
            row[c] = 0.0
    return row[FEATURE_COLS]

def predict_matchup(home_team: str, away_team: str, verbose: bool = True):
    h, a = normalize_team(home_team), normalize_team(away_team)
    row = _row_from_teams(h, a)
    proba_home = float(pipeline.predict_proba(row[FEATURE_COLS])[0, 1])
    proba_away = 1.0 - proba_home
    winner = h if proba_home >= 0.5 else a
    if verbose:
        print(f"\nPrediction ({h} vs {a})")
        print(f" Home win probability: {proba_home:.3f}")
        print(f" Away win probability: {proba_away:.3f}")
        tag = ("💪 Strong favorite" if max(proba_home, proba_away) >= 0.70 else
               "👍 Moderate favorite" if max(proba_home, proba_away) >= 0.60 else
               "🤏 Slight favorite" if max(proba_home, proba_away) >= 0.55 else
               "🎲 Toss-up game")
        print(f" ==> Predicted winner: {winner} ({max(proba_home, proba_away):.1%})  {tag}")
        print("-"*50)
    return {"home": h, "away": a, "p_home": proba_home, "p_away": proba_away, "winner": winner}

def predict(home_team: str, away_team: str):
    return predict_matchup(home_team, away_team, verbose=True)

# ===============================
# TODAY'S SLATE
# ===============================
def predict_todays_games():
    if not SPORTS_BLAZE_KEY:
        print("Set SPORTS_BLAZE_KEY to predict today's games automatically.")
        return
    today = pd.Timestamp.today(tz="America/Chicago").strftime("%Y-%m-%d")
    print(f"Fetching SportsBlaze daily schedule for {today} ...")
    games = _sportsblaze_daily_schedule(today)
    if not games:
        print("No NFL games returned by API for this date.")
        return
    print(f"\nToday's NFL Games ({today}) — Predictions:")
    for i, (home, away) in enumerate(games, 1):
        try:
            res = predict_matchup(home, away, verbose=False)
            print(f"{i}. {res['home']} vs {res['away']}  ->  {res['winner']}  "
                  f"(H:{res['p_home']:.1%} / A:{res['p_away']:.1%})")
        except Exception as e:
            print(f"{i}. {home} vs {away}: could not predict ({e})")

# ===============================
# INTERACTIVE
# ===============================
def interactive_prediction():
    print("\n" + "="*50)
    print("NFL GAME PREDICTOR")
    print("="*50)
    print("Commands:")
    print("  - Enter: Home, Away     (e.g., 'Chiefs, Eagles')")
    print("  - Type 'today' to predict today's games")
    print("  - Type 'exit' to quit")
    print("="*50)
    while True:
        try:
            user_input = input("\n> ").strip()
            if user_input == "" or user_input.lower() in ["exit","quit"]:
                print("Exiting...")
                break
            if user_input.lower() == "today":
                predict_todays_games()
                continue
            teams = [t.strip() for t in user_input.split(",")]
            if len(teams) != 2:
                print("Please enter two teams separated by a comma.")
                continue
            predict_matchup(teams[0], teams[1])
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}. Please try again.")

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    if SPORTS_BLAZE_KEY:
        predict_todays_games()
    interactive_prediction()
