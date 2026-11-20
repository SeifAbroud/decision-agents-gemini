"""Pipeline hors-ligne : données Favorita -> KPIs + prévisions -> artefacts Parquet.

Exécuté UNE fois sur Kaggle (ou Colab / en local). Le dashboard ne lit que le
dossier `artifacts/` produit ici : il n'a jamais besoin de relancer ce code.

    python -m pipeline.build --data /kaggle/input/store-sales-time-series-forecasting --out artifacts
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

HORIZON = 16  # horizon du jeu de test Kaggle (16 jours)


# --------------------------------------------------------------------------- #
# 1. Chargement et préparation
# --------------------------------------------------------------------------- #
def load(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    d = Path(data_dir)
    rd = lambda f, **kw: pd.read_csv(d / f, **kw)
    data = {
        "train": rd("train.csv", parse_dates=["date"],
                    dtype={"store_nbr": "int16", "onpromotion": "float32", "sales": "float32"}),
        "test": rd("test.csv", parse_dates=["date"], dtype={"store_nbr": "int16", "onpromotion": "float32"}),
        "stores": rd("stores.csv"),
        "oil": rd("oil.csv", parse_dates=["date"]),
        "holidays": rd("holidays_events.csv", parse_dates=["date"]),
        "transactions": rd("transactions.csv", parse_dates=["date"]),
    }
    return data


def holiday_calendar(holidays: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par (date, store_nbr) jour férié effectif, en tenant compte du niveau
    national / régional (state) / local (city) et des jours « transferred »."""
    h = holidays.copy()
    h = h[~h["transferred"].astype(str).str.lower().eq("true")]
    h = h[h["type"] != "Work Day"]
    rows = []
    for _, r in h.iterrows():
        if r["locale"] == "National":
            st = stores["store_nbr"]
        elif r["locale"] == "Regional":
            st = stores.loc[stores["state"] == r["locale_name"], "store_nbr"]
        else:
            st = stores.loc[stores["city"] == r["locale_name"], "store_nbr"]
        for s in st:
            rows.append((r["date"], s, r["locale"] == "National", r["description"], r["type"]))
    cal = pd.DataFrame(rows, columns=["date", "store_nbr", "national", "description", "htype"])
    return cal.drop_duplicates(["date", "store_nbr"])


def prepare(data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retourne (panel, hol_cal). panel = train+test enrichi, une ligne par date x store x family."""
    train, test, stores = data["train"], data["test"], data["stores"]
    panel = pd.concat([train.assign(is_test=0), test.assign(is_test=1, sales=np.nan)], ignore_index=True)
    panel = panel.merge(stores[["store_nbr", "city", "state", "type", "cluster"]], on="store_nbr", how="left")

    # pétrole : calendrier complet + forward fill (week-ends et trous)
    oil = data["oil"].set_index("date")["dcoilwtico"]
    full = pd.date_range(panel["date"].min() - pd.Timedelta(days=30), panel["date"].max())
    oil = oil.reindex(full).ffill().bfill().rename("oil").rename_axis("date").reset_index()
    panel = panel.merge(oil, on="date", how="left")

    hol = holiday_calendar(data["holidays"], stores)
    panel = panel.merge(hol[["date", "store_nbr", "national"]].assign(is_holiday=1),
                        on=["date", "store_nbr"], how="left")
    panel["is_holiday"] = panel["is_holiday"].fillna(0).astype("int8")
    panel["is_national_holiday"] = panel["national"].fillna(False).astype("int8")
    panel = panel.drop(columns="national")

    dt = panel["date"].dt
    panel["dow"] = dt.dayofweek.astype("int8")
    panel["dom"] = dt.day.astype("int8")
    panel["month"] = dt.month.astype("int8")
    # paie du secteur public : le 15 et le dernier jour du mois (+ 2 jours suivants)
    panel["payday"] = ((dt.day == 15) | dt.is_month_end | (dt.day <= 2) | dt.day.isin([16, 17])).astype("int8")
    panel = panel.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    return panel, hol


# --------------------------------------------------------------------------- #
# 2. Indicateurs de performance (KPIs)
# --------------------------------------------------------------------------- #
def pct(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(b > 0, (a / b - 1) * 100, np.nan)
    return np.round(r, 1)


def build_kpis(panel: pd.DataFrame, data: dict, hol: pd.DataFrame) -> dict[str, pd.DataFrame | dict]:
    tr = panel[panel.is_test == 0]
    last = tr["date"].max()
    w28 = (tr.date > last - pd.Timedelta(days=28))
    p28 = (tr.date <= last - pd.Timedelta(days=28)) & (tr.date > last - pd.Timedelta(days=56))
    y28 = (tr.date > last - pd.Timedelta(days=364 + 28)) & (tr.date <= last - pd.Timedelta(days=364))
    txn = data["transactions"]

    out: dict = {}
    # séries quotidiennes agrégées
    daily_total = tr.groupby("date", as_index=False).agg(sales=("sales", "sum"), onpromotion=("onpromotion", "sum"))
    daily_total = daily_total.merge(txn.groupby("date", as_index=False)["transactions"].sum(), on="date", how="left")
    out["daily_total"] = daily_total
    daily_store = tr.groupby(["date", "store_nbr"], as_index=False)["sales"].sum().merge(
        txn, on=["date", "store_nbr"], how="left")
    out["daily_store"] = daily_store
    out["daily_family"] = tr.groupby(["date", "family"], as_index=False).agg(
        sales=("sales", "sum"), onpromotion=("onpromotion", "sum"))
    wk = tr.assign(week=tr["date"].dt.to_period("W-SUN").dt.start_time)
    out["weekly_store_family"] = wk.groupby(["week", "store_nbr", "family"], as_index=False)["sales"].sum()

    # KPIs magasins
    st = data["stores"].copy()
    g = lambda m: tr[m].groupby("store_nbr")["sales"].sum()
    st["sales_28d"] = st.store_nbr.map(g(w28)).fillna(0).round(1)
    st["sales_prev_28d"] = st.store_nbr.map(g(p28)).fillna(0).round(1)
    st["sales_ly_28d"] = st.store_nbr.map(g(y28)).fillna(0).round(1)
    st["growth_28d_pct"] = pct(st.sales_28d, st.sales_prev_28d)
    st["yoy_pct"] = pct(st.sales_28d, st.sales_ly_28d)
    t28 = txn[txn.date > last - pd.Timedelta(days=28)].groupby("store_nbr")["transactions"].sum()
    st["transactions_28d"] = st.store_nbr.map(t28).fillna(0)
    st["sales_per_txn"] = np.round(np.where(st.transactions_28d > 0, st.sales_28d / st.transactions_28d.replace(0, np.nan), np.nan), 2)
    st["cluster_median_28d"] = st.groupby("cluster")["sales_28d"].transform("median").round(1)
    st["gap_vs_cluster_pct"] = pct(st.sales_28d, st.cluster_median_28d)
    st["rank_in_cluster"] = st.groupby("cluster")["sales_28d"].rank(ascending=False, method="min").astype(int)
    st["stores_in_cluster"] = st.groupby("cluster")["store_nbr"].transform("count")
    zero_days = (tr[w28].groupby(["store_nbr", "date"])["sales"].sum() == 0).groupby("store_nbr").sum()
    st["zero_sales_days_28d"] = st.store_nbr.map(zero_days).fillna(0).astype(int)
    out["store_kpis"] = st

    # KPIs familles
    fam = pd.DataFrame({"family": sorted(tr.family.unique())})
    gf = lambda m: tr[m].groupby("family")["sales"].sum()
    fam["sales_28d"] = fam.family.map(gf(w28)).fillna(0).round(1)
    fam["sales_prev_28d"] = fam.family.map(gf(p28)).fillna(0).round(1)
    fam["sales_ly_28d"] = fam.family.map(gf(y28)).fillna(0).round(1)
    fam["growth_28d_pct"] = pct(fam.sales_28d, fam.sales_prev_28d)
    fam["yoy_pct"] = pct(fam.sales_28d, fam.sales_ly_28d)
    fam["share_pct"] = (fam.sales_28d / fam.sales_28d.sum() * 100).round(2)
    fam = fam.sort_values("sales_28d", ascending=False)
    cum = fam.share_pct.cumsum()
    fam["abc_class"] = np.where(cum <= 80, "A", np.where(cum <= 95, "B", "C"))
    d_f = out["daily_family"][out["daily_family"].date > last - pd.Timedelta(days=91)]
    cv = d_f.groupby("family")["sales"].agg(lambda s: s.std() / s.mean() if s.mean() > 0 else np.nan)
    fam["volatility_cv"] = fam.family.map(cv).round(2)
    sf28 = tr[w28].groupby(["family", "store_nbr"])["sales"].sum()
    fam["stores_zero_sales_28d"] = fam.family.map((sf28 == 0).groupby("family").sum()).fillna(0).astype(int)
    promo_days = tr[tr.date > last - pd.Timedelta(days=91)].groupby("family")["onpromotion"].apply(lambda s: (s > 0).mean() * 100)
    fam["promo_rows_pct_90d"] = fam.family.map(promo_days).round(1)
    out["family_kpis"] = fam.reset_index(drop=True)

    # Effet promo : ventes en promo vs base hors promo (même magasin, famille, jour de semaine), 365 derniers jours
    y = tr[tr.date > last - pd.Timedelta(days=365)][["store_nbr", "family", "dow", "sales", "onpromotion"]]
    base = y[y.onpromotion == 0].groupby(["store_nbr", "family", "dow"])["sales"].mean().rename("base")
    yp = y[y.onpromotion > 0].join(base, on=["store_nbr", "family", "dow"])
    yp = yp[yp.base > 0]
    yp["ratio"] = yp.sales / yp.base
    pu = yp.groupby("family").agg(promo_obs=("ratio", "size"), uplift=("ratio", "median")).reset_index()
    pu["promo_uplift_pct"] = ((pu.uplift - 1) * 100).round(1)
    out["promo_uplift"] = pu[pu.promo_obs >= 30][["family", "promo_obs", "promo_uplift_pct"]].sort_values(
        "promo_uplift_pct", ascending=False).reset_index(drop=True)

    # Effet jours fériés nationaux : ventes du jour vs moyenne du même jour de semaine (±1, ±2 semaines)
    dt_ = daily_total.set_index("date")["sales"]
    nat = hol[hol.national].drop_duplicates("date")
    rows = []
    for _, r in nat.iterrows():
        d0 = r["date"]
        if d0 not in dt_.index:
            continue
        ref = [d0 + pd.Timedelta(weeks=k) for k in (-2, -1, 1, 2)]
        ref = [dt_.get(x) for x in ref if x in dt_.index]
        if ref and np.nanmean(ref) > 0:
            rows.append((r["description"], d0, float(dt_[d0]), float(np.nanmean(ref))))
    he = pd.DataFrame(rows, columns=["holiday", "date", "sales", "reference_sales"])
    if len(he):
        he["effect_pct"] = pct(he.sales, he.reference_sales)
        he = he.groupby("holiday").agg(occurrences=("date", "size"), effect_pct=("effect_pct", "mean"),
                                       last_date=("date", "max")).reset_index()
        he["effect_pct"] = he.effect_pct.round(1)
    out["holiday_effect"] = he.sort_values("effect_pct") if len(he) else he

    # indicateurs globaux
    pay = tr.groupby(["date", "payday"])["sales"].sum().reset_index()
    pay_eff = pct(pay[pay.payday == 1].sales.mean(), pay[pay.payday == 0].sales.mean())
    wk_tot = daily_total.set_index("date").resample("W")["sales"].sum()
    wk_oil = panel.drop_duplicates("date").set_index("date")["oil"].resample("W").mean()
    oil_corr = float(pd.concat([wk_tot, wk_oil], axis=1).dropna().corr().iloc[0, 1])
    tot28, totp, totly = tr[w28].sales.sum(), tr[p28].sales.sum(), tr[y28].sales.sum()
    out["global"] = {
        "last_date": str(last.date()),
        "first_date": str(tr.date.min().date()),
        "n_stores": int(tr.store_nbr.nunique()),
        "n_families": int(tr.family.nunique()),
        "n_series": int(tr.groupby(["store_nbr", "family"]).ngroups),
        "sales_28d": round(float(tot28), 1),
        "sales_prev_28d": round(float(totp), 1),
        "growth_28d_pct": float(pct(tot28, totp)),
        "yoy_pct": float(pct(tot28, totly)) if totly > 0 else None,
        "avg_daily_sales_28d": round(float(tot28 / 28), 1),
        "transactions_28d": int(txn[txn.date > last - pd.Timedelta(days=28)].transactions.sum()),
        "payday_effect_pct": float(pay_eff),
        "oil_sales_weekly_corr": round(oil_corr, 3),
        "promo_rows_pct_28d": round(float((tr[w28].onpromotion > 0).mean() * 100), 1),
    }
    return out


# --------------------------------------------------------------------------- #
# 3. Prévision (modèle global LightGBM, horizon direct de 16 jours)
# --------------------------------------------------------------------------- #
FEATURES = ["store_nbr", "family_code", "type_code", "cluster", "dow", "dom", "month", "payday",
            "is_holiday", "is_national_holiday", "onpromotion", "oil",
            "lag_16", "lag_21", "lag_28", "lag_35", "lag_42", "lag_364",
            "rmean_7", "rmean_14", "rmean_28", "rmean_56", "rstd_28", "dow_mean_4",
            "promo_rmean_28"]


def add_features(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel
    p["family_code"] = p["family"].astype("category").cat.codes.astype("int16")
    p["type_code"] = p["type"].astype("category").cat.codes.astype("int8")
    y = np.log1p(p["sales"].astype("float32"))
    p["y"] = y
    grp = p.groupby(["store_nbr", "family"], sort=False)["y"]
    for L in (16, 21, 28, 35, 42, 364):
        p[f"lag_{L}"] = grp.shift(L).astype("float32")
    s16 = grp.shift(HORIZON)
    g16 = s16.groupby([p["store_nbr"], p["family"]], sort=False)
    for w in (7, 14, 28, 56):
        p[f"rmean_{w}"] = g16.transform(lambda s: s.rolling(w, min_periods=1).mean()).astype("float32")
    p["rstd_28"] = g16.transform(lambda s: s.rolling(28, min_periods=2).std()).astype("float32")
    p["dow_mean_4"] = p[["lag_21", "lag_28", "lag_35", "lag_42"]].mean(axis=1).astype("float32")
    pr = p.groupby(["store_nbr", "family"], sort=False)["onpromotion"]
    p["promo_rmean_28"] = pr.transform(lambda s: s.shift(1).rolling(28, min_periods=1).mean()).astype("float32")
    return p


def rmsle(y_true, y_pred):
    y_true, y_pred = np.clip(y_true, 0, None), np.clip(y_pred, 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def wape(y_true, y_pred):
    return float(np.sum(np.abs(y_true - y_pred)) / max(np.sum(np.abs(y_true)), 1e-9) * 100)


def train_forecast(panel: pd.DataFrame, train_days: int = 420, rounds: int = 700, log=print):
    import lightgbm as lgb

    p = add_features(panel)
    tr_all = p[p.is_test == 0]
    last = tr_all.date.max()
    val_start = last - pd.Timedelta(days=HORIZON - 1)
    fit_from = val_start - pd.Timedelta(days=train_days)
    fit = tr_all[(tr_all.date >= fit_from) & (tr_all.date < val_start)].dropna(subset=["lag_42"])
    val = tr_all[tr_all.date >= val_start]
    test = p[p.is_test == 1]

    # séries mortes (aucune vente sur 56 jours avant la fenêtre) -> prévision 0
    def dead_mask(df, ref_date):
        recent = tr_all[(tr_all.date > ref_date - pd.Timedelta(days=56)) & (tr_all.date <= ref_date)]
        z = recent.groupby(["store_nbr", "family"])["sales"].sum()
        z = z[z == 0].index
        return pd.MultiIndex.from_frame(df[["store_nbr", "family"]]).isin(z)

    params = dict(objective="regression", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, num_threads=0)
    cat = ["store_nbr", "family_code", "type_code", "cluster"]
    t0 = time.time()
    dtr = lgb.Dataset(fit[FEATURES], fit["y"], categorical_feature=cat, free_raw_data=False)
    dva = lgb.Dataset(val[FEATURES], val["y"], categorical_feature=cat, reference=dtr)
    model = lgb.train(params, dtr, rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    best_it = model.best_iteration or rounds
    log(f"  point model: {best_it} rounds, {time.time() - t0:.0f}s")

    def q_model(alpha, data_):
        qp = dict(params, objective="quantile", alpha=alpha)
        return lgb.train(qp, lgb.Dataset(data_[FEATURES], data_["y"], categorical_feature=cat),
                         max(150, int(best_it * 0.6)))

    # --- validation (backtest sur les 16 derniers jours connus)
    vd = dead_mask(val, val_start - pd.Timedelta(days=1))
    pv = np.expm1(model.predict(val[FEATURES], num_iteration=best_it)).clip(0)
    pv[vd] = 0
    q10v = q_model(0.1, fit)
    q90v = q_model(0.9, fit)
    p10v = np.expm1(q10v.predict(val[FEATURES])).clip(0)
    p90v = np.expm1(q90v.predict(val[FEATURES])).clip(0)
    p10v[vd] = 0
    p90v[vd] = 0
    yv = val["sales"].values
    naive = np.expm1(val["lag_21"].values).clip(0)  # même jour de semaine, 3 semaines avant
    ma28 = np.expm1(val["rmean_28"].values).clip(0)
    naive, ma28 = np.nan_to_num(naive), np.nan_to_num(ma28)
    metrics = {
        "validation_window": [str(val_start.date()), str(last.date())],
        "horizon_days": HORIZON,
        "models": {
            "LightGBM (global)": {"rmsle": round(rmsle(yv, pv), 4), "wape_pct": round(wape(yv, pv), 2)},
            "Naive saisonnier (J-21)": {"rmsle": round(rmsle(yv, naive), 4), "wape_pct": round(wape(yv, naive), 2)},
            "Moyenne mobile 28 j": {"rmsle": round(rmsle(yv, ma28), 4), "wape_pct": round(wape(yv, ma28), 2)},
        },
        "interval_coverage_p10_p90_pct": round(float(np.mean((yv >= p10v) & (yv <= p90v)) * 100), 1),
        "best_iteration": int(best_it),
        "train_rows": int(len(fit)),
        "features": FEATURES,
    }
    validation = val[["date", "store_nbr", "family"]].assign(actual=yv, pred=pv, p10=p10v, p90=p90v,
                                                             naive=naive, ma28=ma28)

    # --- réentraînement sur toutes les données connues puis prévision des 16 jours de test
    t0 = time.time()
    full_fit = tr_all[tr_all.date >= fit_from + pd.Timedelta(days=HORIZON)].dropna(subset=["lag_42"])
    final = lgb.train(params, lgb.Dataset(full_fit[FEATURES], full_fit["y"], categorical_feature=cat), best_it)
    td = dead_mask(test, last)
    pt = np.expm1(final.predict(test[FEATURES])).clip(0)
    p10t = np.expm1(q_model(0.1, full_fit).predict(test[FEATURES])).clip(0)
    p90t = np.expm1(q_model(0.9, full_fit).predict(test[FEATURES])).clip(0)
    for a in (pt, p10t, p90t):
        a[td] = 0
    p10t, p90t = np.minimum(p10t, pt), np.maximum(p90t, pt)
    log(f"  final + quantile models: {time.time() - t0:.0f}s")
    forecast = test[["id", "date", "store_nbr", "family", "onpromotion"]].assign(p50=pt, p10=p10t, p90=p90t)
    # Le modèle optimise le RMSLE (cible log) : la médiane sous-estime les totaux.
    # On calibre un facteur par famille sur la validation pour les synthèses « business ».
    calib = validation.groupby("family").apply(
        lambda d: d.actual.sum() / d.pred.sum() if d.pred.sum() > 0 else 1.0, include_groups=False).clip(0.8, 1.3)
    fac = forecast.family.map(calib).fillna(1.0).values
    forecast["mean"] = forecast.p50 * fac
    forecast["p10"], forecast["p90"] = forecast.p10 * fac, forecast.p90 * fac
    metrics["bias_calibration_by_family"] = {k: round(float(v), 3) for k, v in calib.items()}

    fi = pd.DataFrame({"feature": FEATURES, "gain": final.feature_importance("gain")})
    fi["gain_pct"] = (fi.gain / fi.gain.sum() * 100).round(2)
    fi = fi.sort_values("gain_pct", ascending=False)[["feature", "gain_pct"]]
    return forecast, validation, metrics, fi


def agg_band(fc: pd.DataFrame, key) -> pd.DataFrame:
    """Agrège des prévisions en gardant un intervalle cohérent.
    Les quantiles ne s'additionnent pas : on combine les demi-largeurs en supposant
    l'indépendance des erreurs (racine de la somme des carrés)."""
    d = fc.assign(lo2=(fc["mean"] - fc.p10).clip(lower=0) ** 2, hi2=(fc.p90 - fc["mean"]).clip(lower=0) ** 2)
    g = d.groupby(key).agg(mean=("mean", "sum"), lo2=("lo2", "sum"), hi2=("hi2", "sum")).reset_index()
    g["p10"] = (g["mean"] - np.sqrt(g.lo2)).clip(lower=0)
    g["p90"] = g["mean"] + np.sqrt(g.hi2)
    return g.drop(columns=["lo2", "hi2"])


def forecast_summaries(forecast: pd.DataFrame, panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tr = panel[panel.is_test == 0]
    last = tr.date.max()
    recent = tr[tr.date > last - pd.Timedelta(days=HORIZON)]
    ly_start = forecast.date.min() - pd.Timedelta(days=364)
    ly = tr[(tr.date >= ly_start) & (tr.date < ly_start + pd.Timedelta(days=HORIZON))]
    out = {}
    for key in ("store_nbr", "family"):
        f = agg_band(forecast, key).rename(columns={"mean": "forecast_16d", "p10": "p10_16d", "p90": "p90_16d"})
        f["promo_items"] = f[key].map(forecast.groupby(key)["onpromotion"].sum())
        f["last_16d"] = f[key].map(recent.groupby(key)["sales"].sum())
        f["ly_same_16d"] = f[key].map(ly.groupby(key)["sales"].sum())
        f["change_vs_last_16d_pct"] = pct(f.forecast_16d, f.last_16d)
        f["change_vs_ly_pct"] = pct(f.forecast_16d, f.ly_same_16d)
        f["uncertainty_pct"] = np.round(np.where(f.forecast_16d > 0, (f.p90_16d - f.p10_16d) / f.forecast_16d * 100, np.nan), 1)
        for c in ("forecast_16d", "p10_16d", "p90_16d", "last_16d", "ly_same_16d"):
            f[c] = f[c].round(1)
        out[f"forecast_by_{'store' if key == 'store_nbr' else 'family'}"] = f
    daily = agg_band(forecast, "date")
    out["forecast_daily_total"] = daily
    return out


# --------------------------------------------------------------------------- #
# 4. Orchestration + export
# --------------------------------------------------------------------------- #
def run(data_dir: str | Path, out_dir: str | Path = "artifacts", source: str = "kaggle",
        train_days: int = 420, log=print) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log("1/4 Chargement des données…")
    data = load(data_dir)
    panel, hol = prepare(data)
    log(f"    panel: {len(panel):,} lignes")
    log("2/4 Calcul des KPIs…")
    k = build_kpis(panel, data, hol)
    log("3/4 Entraînement du modèle de prévision…")
    forecast, validation, metrics, fi = train_forecast(panel, train_days=train_days, log=log)
    fs = forecast_summaries(forecast, panel)
    log("4/4 Export des artefacts…")

    g = k.pop("global")
    for name, df in {**k, **fs}.items():
        df.to_parquet(out / f"{name}.parquet", index=False)
    forecast.drop(columns="id").to_parquet(out / "forecast.parquet", index=False)
    validation.to_parquet(out / "validation.parquet", index=False)
    fi.to_parquet(out / "feature_importance.parquet", index=False)
    last = panel[panel.is_test == 0].date.max()
    hist = panel[(panel.is_test == 0) & (panel.date > last - pd.Timedelta(days=112))][
        ["date", "store_nbr", "family", "sales", "onpromotion"]]
    hist.to_parquet(out / "history_recent.parquet", index=False)
    data["stores"].to_parquet(out / "stores.parquet", index=False)
    upcoming = data["holidays"][data["holidays"].date.between(forecast.date.min(), forecast.date.max())]
    upcoming.to_parquet(out / "holidays_upcoming.parquet", index=False)
    if source == "kaggle":
        forecast[["id", "p50"]].rename(columns={"p50": "sales"}).to_csv(out / "submission.csv", index=False)
    meta = {**g, "source": source, "forecast_start": str(forecast.date.min().date()),
            "forecast_end": str(forecast.date.max().date()), "built_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            "build_seconds": round(time.time() - t0)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    log(f"Terminé en {time.time() - t0:.0f}s -> {out.resolve()}")
    log(json.dumps(metrics["models"], indent=2, ensure_ascii=False))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dossier contenant train.csv, test.csv, …")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--source", default="kaggle", choices=["kaggle", "synthetic"])
    ap.add_argument("--train-days", type=int, default=420)
    a = ap.parse_args()
    run(a.data, a.out, a.source, a.train_days)
