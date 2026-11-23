"""Download NBA player game logs and load them into SQLite."""

import sqlite3
import time
from contextlib import closing, contextmanager
from datetime import date
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
from nba_api.stats.endpoints.leaguegamelog import LeagueGameLog

FIRST_SEASON = 2010
TODAY = date.today()
# October is the usual start of the next NBA season.
CURRENT_SEASON = TODAY.year if TODAY.month >= 10 else TODAY.year - 1
LAST_SEASON = CURRENT_SEASON

DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_DIR = DATA_DIR / "raw"
DB_PATH = DATA_DIR / "nba.db"
SEASON_TYPES = {"2": "Regular Season", "4": "Playoffs"}

# Rename the API abbreviations and give SQLite stable column types.
COLUMNS = {
    "PLAYER_ID": ("player_id", "INTEGER"),
    "PLAYER_NAME": ("player_name", "TEXT"),
    "TEAM_ID": ("team_id", "INTEGER"),
    "TEAM_ABBREVIATION": ("team_abbreviation", "TEXT"),
    "GAME_ID": ("game_id", "TEXT"),
    "GAME_DATE": ("game_date", "TEXT"),
    "MATCHUP": ("matchup", "TEXT"),
    "WL": ("win_loss", "TEXT"),
    "MIN": ("minutes", "REAL"),
    "PTS": ("points", "INTEGER"),
    "REB": ("rebounds", "INTEGER"),
    "AST": ("assists", "INTEGER"),
    "STL": ("steals", "INTEGER"),
    "BLK": ("blocks", "INTEGER"),
    "TOV": ("turnovers", "INTEGER"),
    "FGM": ("field_goals_made", "INTEGER"),
    "FGA": ("field_goals_attempted", "INTEGER"),
    "FG_PCT": ("field_goal_percentage", "REAL"),
    "FG3M": ("three_pointers_made", "INTEGER"),
    "FG3A": ("three_pointers_attempted", "INTEGER"),
    "FG3_PCT": ("three_point_percentage", "REAL"),
    "FTM": ("free_throws_made", "INTEGER"),
    "FTA": ("free_throws_attempted", "INTEGER"),
    "FT_PCT": ("free_throw_percentage", "REAL"),
}
SQL_TYPES = dict(COLUMNS.values(), season="TEXT", season_type="TEXT")


@contextmanager
def staged_file(destination):
    # Write next to the destination so a failed download or build leaves the old file intact.
    with NamedTemporaryFile(dir=destination.parent, suffix=".tmp", delete=False) as file:
        path = Path(file.name)
    try:
        yield path
        path.replace(destination)
    finally:
        path.unlink(missing_ok=True)


def prepare_games(raw):
    """Give API exports and local CSVs the same database columns."""
    if "SEASON_ID" in raw:
        codes = raw["SEASON_ID"].astype("string")
        years = codes.str[1:]
        end_years = ((pd.to_numeric(years) + 1) % 100).astype("string").str.zfill(2)
        raw = raw.assign(
            season=years + "-" + end_years,
            season_type=codes.str[0].map(SEASON_TYPES),
        )

    games = raw.rename(columns={source: name for source, (name, _) in COLUMNS.items()})
    missing = SQL_TYPES.keys() - set(games.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")
    games = games[list(SQL_TYPES)].copy()

    # Older logs can have a clock like 32:30. Store it as 32.5 minutes for SQL.
    minutes = games["minutes"].astype("string")
    clock = minutes.str.contains(":", na=False)
    seconds = pd.to_timedelta("00:" + minutes.loc[clock]).dt.total_seconds()
    minutes.loc[clock] = (seconds / 60).astype("string")
    games["minutes"] = minutes
    for column, sql_type in SQL_TYPES.items():
        if sql_type == "TEXT":
            games[column] = games[column].astype("string").str.strip().replace("", pd.NA)
        else:
            games[column] = pd.to_numeric(games[column])

    required = ["player_id", "player_name", "team_id", "game_id", "game_date", "season", "season_type"]
    if games[required].isna().any().any():
        raise ValueError("Every row needs a player, team ID, game ID, date, season, and season type.")
    valid_seasons = games["season"].str.fullmatch(r"\d{4}-\d{2}").all()
    valid_season_types = games["season_type"].isin(SEASON_TYPES.values()).all()
    if not (valid_seasons and valid_season_types):
        raise ValueError("Use season labels like 2023-24 and season types Regular Season or Playoffs.")
    games["game_date"] = pd.to_datetime(games["game_date"], format="%Y-%m-%d").dt.strftime("%Y-%m-%d")
    return games


def load_season(year, season_type):
    season = f"{year}-{(year + 1) % 100:02d}"
    kind = season_type.lower().replace(" ", "_")
    # A partial cache must not be mistaken for a completed season next year.
    suffix = ".in_progress.csv" if year == CURRENT_SEASON else ".csv"
    path = RAW_DIR / f"{season}_{kind}{suffix}"

    if path.exists():
        print(f"Using {path.name}", flush=True)
        return prepare_games(pd.read_csv(path, dtype=str))

    print(f"Downloading {season} {season_type}", flush=True)
    for attempt in range(1, 4):
        time.sleep(attempt)
        try:
            raw = LeagueGameLog(
                player_or_team_abbreviation="P",
                season=season,
                season_type_all_star=season_type,
                timeout=20,
            ).get_data_frames()[0]
            if raw.empty and year == CURRENT_SEASON:
                # The API can return an empty dataset before games are played.
                raw = pd.DataFrame(columns=["SEASON_ID", *COLUMNS])
            games = prepare_games(raw)
            if games.empty and year != CURRENT_SEASON:
                raise ValueError("The API returned no games for a completed season.")
            if not (games["season"].eq(season) & games["season_type"].eq(season_type)).all():
                raise ValueError("The API returned games for another season or season type.")
        except Exception as error:
            print(f"  Attempt {attempt}/3 failed: {error}", flush=True)
            if attempt == 3:
                raise RuntimeError(f"Could not download {season} {season_type}. Rerun to resume.") from error
        else:
            with staged_file(path) as staged:
                raw.to_csv(staged, index=False)
            return games


def game_batches():
    # A top-level CSV replaces the download so an offline build is possible.
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if csv_files:
        if len(csv_files) > 1:
            raise ValueError("Keep one fallback CSV directly in data/.")
        print(f"Loading {csv_files[0].name} instead of downloading.")
        yield prepare_games(pd.read_csv(csv_files[0], dtype=str))
        return

    if FIRST_SEASON > LAST_SEASON:
        raise ValueError("FIRST_SEASON must be at or before LAST_SEASON.")
    RAW_DIR.mkdir(exist_ok=True)
    for year in range(FIRST_SEASON, LAST_SEASON + 1):
        for season_type in SEASON_TYPES.values():
            yield load_season(year, season_type)


def build_database():
    DATA_DIR.mkdir(exist_ok=True)
    with staged_file(DB_PATH) as staged:
        with closing(sqlite3.connect(staged)) as db:
            for games in game_batches():
                games.to_sql("player_games", db, if_exists="append", index=False, dtype=SQL_TYPES)

            db.executescript("""
                CREATE UNIQUE INDEX player_game ON player_games (player_id, game_id);
                CREATE INDEX player_name ON player_games (player_name COLLATE NOCASE);
                CREATE INDEX season ON player_games (season, season_type);
                CREATE INDEX game_date ON player_games (game_date);
            """)
            count, first, last = db.execute("""
                SELECT COUNT(*), MIN(game_date), MAX(game_date) FROM player_games
            """).fetchone()
            if count == 0 or db.execute("PRAGMA quick_check").fetchone() != ("ok",):
                raise ValueError("The database is empty or failed SQLite's integrity check.")

    print(f"player_games: {count:,} rows, {first} through {last}")
    print(f"Saved {DB_PATH}")


if __name__ == "__main__":
    try:
        build_database()
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as error:
        raise SystemExit(f"Database build failed: {error}") from error
