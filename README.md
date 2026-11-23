# nba-stats-agent

Ask questions about NBA player stats in everyday language. The agent looks up the answers in game logs. It is for basketball fans and analysts who want to explore scores, trends, and comparisons.

**Question → agent → SQL → SQLite → answer**

`build_db.py` loads the player game logs from NBA Stats. `agent.py` gives the model tools to inspect the database and check and run SQL. `app.py` displays the answer in a local Gradio chat. [NBA Stats endpoint documentation](https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/endpoints/leaguegamelog.md)

## 1. Install dependencies with uv

This project uses uv and requires Python 3.11 or newer. Install uv, then run `uv sync` from the cloned project folder. uv creates the environment and installs the versions in `uv.lock`. [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)

```sh
uv sync
```

## 2. Build the local database

Every clone generates its own `data/nba.db`. Run the builder to download player game logs from NBA Stats and create the SQLite file.

```sh
uv run build_db.py
```

The first download takes a while. The script prints the row count and game date range when it finishes. If a download fails, rerun the command to resume from the season CSVs cached in `data/raw/`.

`FIRST_SEASON` and `LAST_SEASON` at the top of `build_db.py` control which years to download. To refresh the current season, remove its `.in_progress.csv` files in `data/raw/`, rebuild the database, and restart the chat.

For an offline build, put one CSV directly in `data/`. Use a player export with NBA columns and `SEASON_ID`, such as a file from `data/raw/`. A CSV with the `player_games` column names defined in `build_db.py` also works. Include `season` and `season_type`, and format dates as `YYYY-MM-DD`. Move the fallback CSV out of `data/` to resume NBA downloads.

## 3. Set your OpenAI API key

Create a local file named `.env` next to `pyproject.toml` and add this line with your own OpenAI API key.

```dotenv
OPENAI_API_KEY=your_openai_api_key
```

The default model is `gpt-4o-mini`. Edit `MODEL` at the top of `agent.py` to use a different model.

## 4. Run the app

```sh
uv run app.py
```

Open `http://127.0.0.1:7860` and try an example. The **Thinking** tab shows tool activity live, including SQL checks and database results. Open **SQL used** to see the queries the agent ran.

## How accurate is it?

Run the evaluation with the standard NBA download and your API key.

```sh
uv run eval/run_eval.py
```

The script calculates the expected results for the questions in `eval/questions.csv`, prints an accuracy score, and writes `eval/results.csv`. It looks for the expected value in the answer or an explanation that the data is unavailable. Read the notes to review individual grades.

## Design decisions

- SQLite keeps the data in one file and handles this dataset quickly. The SQL toolkit uses a database connection. Switching databases later would involve updating the connection and any SQLite-specific queries.
- The agent follows LangChain's SQL workflow and runs on LangGraph through `create_agent`. [Build a SQL agent (LangChain tutorial)](https://docs.langchain.com/oss/python/langchain/sql-agent)
- The project uses `SQLDatabase` and four tools from `SQLDatabaseToolkit`, which come from the archived `langchain-community` package. [Package archive notice](https://github.com/langchain-ai/langchain-community/issues/674)

## Limitations and safety

- The database connection is read-only. The agent can still generate a wrong query or misinterpret the rows it gets back.
- The standard NBA download starts with 2010-11. The active season's totals and averages reflect games played through the last date in the database.
