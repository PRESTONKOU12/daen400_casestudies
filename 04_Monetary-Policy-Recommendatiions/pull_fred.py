"""
Case Study 4 - pull ONLY the data needed for the four questions.

Setup: .env file next to this script containing   FRED_API_KEY=your_key
Run:   python get_core_data.py

Writes 3 files to ./data/  (delete any old data/ folder first):
  core_monthly.csv      one row per month (2015+), readable column names
  fomc_projections.csv  the FOMC's own median forecasts (benchmarks for Q4)
  data_dictionary.csv   what every column in core_monthly.csv means
"""
import json, os, time, urllib.error, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "data"; OUT.mkdir(exist_ok=True)
START, BASE = "2015-01-01", "2019-12-01"


# ------------------------------------------------------------------ API key from .env
def load_key() -> str:
    for env in [HERE / ".env", Path.cwd() / ".env"]:
        if env.is_file():
            for line in env.read_text().splitlines():
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip().removeprefix("export ").strip(),
                                          v.strip().strip('"').strip("'"))
    key = os.environ.get("FRED_API_KEY", "").strip()
    if not key:
        raise SystemExit("Put FRED_API_KEY=your_key in a .env file next to this script.")
    return key


KEY = load_key()

# ------------------------------------------------------------------ the 20 series we use
# FRED id: (column name, question, plain-English meaning)
SERIES = {
    # Q1 - How tight is policy right now?
    "FEDFUNDS":       ("fed_funds_rate",       "Q1", "Fed's policy interest rate, % - the thing you recommend changing"),
    "PCEPI":          ("pce_index",            "Q1", "Headline PCE price index - the Fed's official 2% inflation target is on this"),
    "PCEPILFE":       ("core_pce_index",       "Q1", "Core PCE price index (no food/energy) - what the Fed actually watches"),
    "UNRATE":         ("unemployment_rate",    "Q1", "Unemployment rate, %"),
    "NROU":           ("natural_unemployment", "Q1", "CBO estimate of 'normal' unemployment, % (quarterly, filled forward)"),
    # Q2 - Did tightening reach autos?  (read top to bottom = the transmission chain)
    "GS5":            ("treasury_5y",          "Q2", "Step 1: 5-year Treasury yield, % - market rate matching a car-loan's length"),
    "TERMCBAUTO48NS": ("auto_loan_rate",       "Q2", "Step 2: bank rate on a 48-month new-car loan, %"),
    "STDSAUTO":       ("auto_lending_tighter", "Q2", "Step 3: net % of banks tightening auto-loan standards (>0 = tighter) (quarterly)"),
    "CARACBW027SBOG": ("auto_loans_bn",        "Q2", "Step 4: auto loans held by banks, $ billions"),
    "TOTALSA":        ("vehicle_sales_mn",     "Q2", "Step 5: vehicles sold, millions/yr (units, so 2021 price spike doesn't distort it)"),
    "IPG3361T3S":     ("auto_production_idx",  "Q2", "Step 6: auto & parts production index (2017=100)"),
    "CUSR0000SETA02": ("used_car_price_idx",   "Q2", "Control: used-car CPI - its 2021 spike marks the chip-shortage supply shock"),
    # Q3 - Did it spread to wider credit and manufacturing?
    "DRTSCILM":       ("business_lending_tighter", "Q3", "Net % of banks tightening business-loan standards (quarterly)"),
    "BUSLOANS":       ("business_loans_bn",    "Q3", "Commercial & industrial loans at banks, $ billions"),
    "DRCLACBS":       ("consumer_delinquency", "Q3", "Consumer-loan delinquency rate at banks, % (quarterly)"),
    "IPMAN":          ("manufacturing_idx",    "Q3", "Manufacturing production index (2017=100)"),
    # Q4 - Forecast inputs
    "DCOILWTICO":     ("oil_price",            "Q4", "WTI crude oil, $/barrel (monthly avg) - how wars hit inflation; ARIMAX input"),
}
FOMC = {  # annual projections, kept in their own file
    "FEDTARMD": "fomc_median_rate", "JCXFEMD": "fomc_median_core_pce", "UNRATEMD": "fomc_median_unemployment",
}

DERIVED = {   # column: (question, meaning)
    "headline_inflation":   ("Q1", "PCE inflation, % change vs. 12 months earlier"),
    "core_inflation":       ("Q1", "Core PCE inflation, % change vs. 12 months earlier"),
    "real_fed_funds":       ("Q1", "fed_funds_rate minus core_inflation (>0 means policy is restrictive)"),
    "unemployment_gap":     ("Q1", "natural_unemployment minus unemployment_rate (>0 = labor market hot)"),
    "taylor_rule_rate":     ("Q1", "Rate the Taylor (1993) rule says the Fed should set, %"),
    "policy_vs_taylor":     ("Q1", "fed_funds_rate minus taylor_rule_rate (<0 = looser than rule)"),
    "used_car_inflation":   ("Q2", "Used-car prices, % change vs. 12 months earlier"),
    "auto_loan_spread":     ("Q2", "auto_loan_rate minus treasury_5y - what banks add on top of market rates"),
    "regime":               ("all", "Policy period: pre-COVID / COVID-easy / hiking / hold at peak / easing / hold (oil shock) / re-tightening"),
}


# ------------------------------------------------------------------ download
def fetch(sid: str) -> pd.Series:
    url = (f"https://api.stlouisfed.org/fred/series/observations?series_id={sid}"
           f"&api_key={KEY}&file_type=json&observation_start={START}")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}),
                                        timeout=30) as r:
                obs = json.loads(r.read())["observations"]
            s = pd.to_numeric(pd.DataFrame(obs)["value"], errors="coerce")
            s.index = pd.to_datetime([o["date"] for o in obs])
            return s.dropna()
        except urllib.error.HTTPError as e:
            if e.code in (400, 404):
                raise
        except Exception:
            pass
        time.sleep(2 ** attempt)
    raise RuntimeError(f"could not download {sid}")


def to_monthly(s: pd.Series) -> pd.Series:
    gap = s.index[-24:].to_series().diff().dt.days.median()
    if gap <= 8:                     # daily/weekly -> monthly average
        return s.resample("MS").mean()
    if gap <= 31:                    # already monthly
        return s.resample("MS").last()
    return s.resample("MS").ffill(limit=2)   # quarterly -> carry value through the quarter


def main():
    cols, failed = {}, []
    for sid, (name, q, _) in SERIES.items():
        try:
            cols[name] = to_monthly(fetch(sid)); print(f"ok      {q}  {name:26s} ({sid})")
        except Exception as e:
            failed.append(sid); print(f"FAILED  {q}  {name:26s} ({sid}): {e}")
        time.sleep(0.5)

    df = pd.DataFrame(cols).loc[:pd.Timestamp.today()]

    # derived columns
    yoy = lambda c: 100 * (df[c] / df[c].shift(12) - 1)
    df["headline_inflation"] = yoy("pce_index")
    df["core_inflation"] = yoy("core_pce_index")
    df["real_fed_funds"] = df.fed_funds_rate - df.core_inflation
    df["unemployment_gap"] = df.natural_unemployment - df.unemployment_rate
    df["taylor_rule_rate"] = 1.0 + df.core_inflation + 0.5 * (df.core_inflation - 2) + df.unemployment_gap
    df["policy_vs_taylor"] = df.fed_funds_rate - df.taylor_rule_rate
    df["used_car_inflation"] = yoy("used_car_price_idx")
    df["auto_loan_spread"] = df.auto_loan_rate - df.treasury_5y
    df["regime"] = pd.cut(df.index, pd.to_datetime(["1900-01-01", "2020-03-01", "2022-03-01", "2023-08-01",
                                                    "2024-09-01", "2026-01-01", "2026-09-01", "2100-01-01"]),
                          labels=["pre-COVID", "COVID-easy", "hiking", "hold at peak", "easing",
                                  "hold (oil shock)", "re-tightening"], right=False)
    df.to_csv(OUT / "core_monthly.csv", index_label="date")

    # FOMC projections
    fomc = {}
    for sid, name in FOMC.items():
        try:
            s = fetch(sid); s.index = s.index.year; fomc[name] = s
        except Exception as e:
            failed.append(sid); print(f"FAILED  {sid}: {e}")
    pd.DataFrame(fomc).to_csv(OUT / "fomc_projections.csv", index_label="year")

    # dictionary
    rows = [(name, q, sid, desc) for sid, (name, q, desc) in SERIES.items()]
    rows += [(c, q, "derived", d) for c, (q, d) in DERIVED.items()]
    rows += [(n, "Q4", s, "FOMC median projection (annual) - benchmark for your forecast") for s, n in FOMC.items()]
    pd.DataFrame(rows, columns=["column", "question", "fred_id", "meaning"]).to_csv(
        OUT / "data_dictionary.csv", index=False)

    print(f"\ncore_monthly.csv: {df.shape[0]} months x {df.shape[1]} columns "
          f"({df.index.min():%Y-%m} to {df.index.max():%Y-%m})")
    print("FAILED:", failed if failed else "none")


if __name__ == "__main__":
    main()