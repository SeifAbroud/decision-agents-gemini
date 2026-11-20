"""Génère des fichiers CSV *synthétiques* avec exactement le même schéma que
le jeu Kaggle « Store Sales - Time Series Forecasting » (Favorita).

Usage :  python -m pipeline.synth --out data/synthetic --start 2016-01-01

Sert uniquement à tester le pipeline et le dashboard sans les vraies données.
Sur Kaggle, le pipeline lit directement /kaggle/input/store-sales-time-series-forecasting.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FAMILIES = [
    "AUTOMOTIVE", "BABY CARE", "BEAUTY", "BEVERAGES", "BOOKS", "BREAD/BAKERY",
    "CELEBRATION", "CLEANING", "DAIRY", "DELI", "EGGS", "FROZEN FOODS",
    "GROCERY I", "GROCERY II", "HARDWARE", "HOME AND KITCHEN I",
    "HOME AND KITCHEN II", "HOME APPLIANCES", "HOME CARE", "LADIESWEAR",
    "LAWN AND GARDEN", "LINGERIE", "LIQUOR,WINE,BEER", "MAGAZINES", "MEATS",
    "PERSONAL CARE", "PET SUPPLIES", "PLAYERS AND ELECTRONICS", "POULTRY",
    "PREPARED FOODS", "PRODUCE", "SCHOOL AND OFFICE SUPPLIES", "SEAFOOD",
]
CITIES = [("Quito", "Pichincha"), ("Guayaquil", "Guayas"), ("Cuenca", "Azuay"),
          ("Santo Domingo", "Santo Domingo de los Tsachilas"), ("Ambato", "Tungurahua"),
          ("Machala", "El Oro"), ("Manta", "Manabi"), ("Loja", "Loja")]


def generate(out: Path, start="2016-01-01", end="2017-08-15", n_stores=54, seed=7):
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range(start, end, freq="D")
    test_dates = pd.date_range(pd.Timestamp(end) + pd.Timedelta(days=1), periods=16, freq="D")

    # stores.csv
    stores = pd.DataFrame({
        "store_nbr": np.arange(1, n_stores + 1),
        "city": [CITIES[i % len(CITIES)][0] for i in range(n_stores)],
        "state": [CITIES[i % len(CITIES)][1] for i in range(n_stores)],
        "type": rng.choice(list("ABCDE"), n_stores, p=[.15, .15, .25, .3, .15]),
        "cluster": rng.integers(1, 18, n_stores),
    })
    stores.to_csv(out / "stores.csv", index=False)

    # oil.csv (jours ouvrés, quelques NaN comme le vrai fichier)
    all_days = dates.append(test_dates)
    bdays = all_days[all_days.dayofweek < 5]
    oil = 45 + np.cumsum(rng.normal(0, 0.6, len(bdays)))
    oil_df = pd.DataFrame({"date": bdays, "dcoilwtico": oil.round(2)})
    oil_df.loc[rng.random(len(oil_df)) < 0.03, "dcoilwtico"] = np.nan
    oil_df.to_csv(out / "oil.csv", index=False)

    # holidays_events.csv
    hol_rows = []
    for y in sorted(set(all_days.year)):
        for md, desc in [("01-01", "Primer dia del ano"), ("05-01", "Dia del Trabajo"),
                         ("08-10", "Primer Grito de Independencia"), ("11-02", "Dia de Difuntos"),
                         ("12-25", "Navidad"), ("12-24", "Navidad-1")]:
            hol_rows.append((f"{y}-{md}", "Holiday", "National", "Ecuador", desc, False))
        hol_rows.append((f"{y}-07-25", "Holiday", "Local", "Guayaquil", "Fundacion de Guayaquil", False))
        hol_rows.append((f"{y}-12-06", "Holiday", "Local", "Quito", "Fundacion de Quito", False))
    hol_rows.append(("2016-04-16", "Event", "National", "Ecuador", "Terremoto Manabi", False))
    hol = pd.DataFrame(hol_rows, columns=["date", "type", "locale", "locale_name", "description", "transferred"])
    hol.to_csv(out / "holidays_events.csv", index=False)
    hol_nat = set(pd.to_datetime(hol.loc[hol.locale == "National", "date"]))

    # niveau de base par (store, family)
    store_scale = rng.lognormal(0, 0.5, n_stores)
    fam_scale = rng.lognormal(2.5, 1.6, len(FAMILIES))
    fam_promo = rng.uniform(0.02, 0.45, len(FAMILIES))  # uplift promo par famille
    dow_effect = np.array([0.95, 0.9, 0.92, 0.93, 1.02, 1.25, 1.18])

    def make(dates_, with_sales):
        n_d, n_s, n_f = len(dates_), n_stores, len(FAMILIES)
        d = np.repeat(dates_.values, n_s * n_f)
        s = np.tile(np.repeat(np.arange(1, n_s + 1), n_f), n_d)
        f = np.tile(np.arange(n_f), n_d * n_s)
        promo_rate = np.where(fam_scale[f] > 30, 0.35, 0.05)
        onpromo = (rng.random(len(d)) < promo_rate) * rng.integers(1, 40, len(d))
        frame = pd.DataFrame({"date": d, "store_nbr": s, "family": np.array(FAMILIES)[f], "onpromotion": onpromo})
        if with_sales:
            dd = pd.DatetimeIndex(d)
            t = (dd - pd.Timestamp(start)).days.values / 365
            payday = ((dd.day == 15) | dd.is_month_end | (dd.day <= 2)).astype(float)
            hol_eff = dd.isin(list(hol_nat)).astype(float)
            dec = (dd.month == 12).astype(float)
            mu = (store_scale[s - 1] * fam_scale[f] * dow_effect[dd.dayofweek.values]
                  * (1 + 0.08 * t) * (1 + 0.12 * payday) * (1 + 0.3 * dec) * (1 - 0.5 * hol_eff)
                  * (1 + fam_promo[f] * (onpromo > 0)))
            sales = rng.gamma(4, mu / 4)
            dead = rng.random((n_s, n_f)) < 0.05  # familles jamais vendues dans certains magasins
            sales[dead[s - 1, f]] = 0
            closed = rng.random(len(d)) < 0.002
            sales[closed] = 0
            frame["sales"] = np.round(sales, 3)
        return frame

    train = make(dates, True)
    train.insert(0, "id", np.arange(len(train)))
    train[["id", "date", "store_nbr", "family", "sales", "onpromotion"]].to_csv(out / "train.csv", index=False)
    test = make(test_dates, False)
    test.insert(0, "id", np.arange(len(train), len(train) + len(test)))
    test.to_csv(out / "test.csv", index=False)

    tr = train.groupby(["date", "store_nbr"], as_index=False)["sales"].sum()
    tr["transactions"] = (tr["sales"] / rng.uniform(8, 14, len(tr))).round().astype(int).clip(lower=0)
    tr.loc[tr.transactions > 0, ["date", "store_nbr", "transactions"]].to_csv(out / "transactions.csv", index=False)
    print(f"Synthetic Favorita data written to {out}  (train rows: {len(train):,})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--stores", type=int, default=54)
    a = ap.parse_args()
    generate(Path(a.out), start=a.start, n_stores=a.stores)
