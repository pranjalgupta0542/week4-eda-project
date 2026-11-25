"""Week 4: EDA + feature engineering on simulated, already-cleaned beauty review data."""
import json, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

df = pd.read_csv("eda_reviews.csv", parse_dates=["review_date"])
CUT, END = pd.Timestamp("2023-09-30"), pd.Timestamp("2023-12-31")
M = {"versions": {"pandas": pd.__version__, "numpy": np.__version__, "scipy": __import__("scipy").__version__,
                  "sklearn": sklearn.__version__, "seaborn": sns.__version__, "matplotlib": matplotlib.__version__}}
sns.set_theme(style="whitegrid", font_scale=0.85)

# ---------------- 1. overview and descriptives ----------------
M["overview"] = {"rows": len(df), "users": int(df.user_id.nunique()), "products": int(df.product_id.nunique()),
                 "start": str(df.review_date.min().date()), "end": str(df.review_date.max().date())}
per_user = df.groupby("user_id").size()
M["desc"] = {"rating_mean": df.rating.mean(), "rating_sd": df.rating.std(), "rating_median": df.rating.median(),
             "pct5": (df.rating == 5).mean(), "pct_neg": (df.rating <= 2).mean(),
             "price_median": df.price.median(), "price_q1": df.price.quantile(.25), "price_q3": df.price.quantile(.75),
             "price_skew": stats.skew(df.price), "words_median": df.review_words.median(), "words_mean": df.review_words.mean(),
             "rev_user_median": per_user.median(), "rev_user_q3": per_user.quantile(.75), "rev_user_max": int(per_user.max()),
             "rev_user_skew": stats.skew(per_user), "rating_by_cat": df.groupby("category").rating.mean().round(2).to_dict()}

# ---------------- 2. statistical tests (review level) ----------------
tests = []
def add(name, planted, stat, p, eff, eff_name, note=""):
    tests.append({"test": name, "planted": planted, "statistic": float(stat), "p": float(p), "effect": float(eff), "effect_name": eff_name, "note": note})
def mwu(a, b):
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided"); return u, p, 2 * u / (len(a) * len(b)) - 1   # rank-biserial
def cramers(tab):
    chi2, p, _, _ = stats.chi2_contingency(tab); n = tab.to_numpy().sum()
    return chi2, p, np.sqrt(chi2 / (n * (min(tab.shape) - 1)))

known = df[df.skin_type != "Unknown"]
H, p = stats.kruskal(*[g.rating for _, g in known.groupby("skin_type")])
eps = (H - 4) / (len(known) - 5)
add("Rating differs by skin type (Kruskal-Wallis)", "partly", H, p, eps, "epsilon-squared", "only Sensitive x fragrance was planted")
sens = known[known.skin_type == "Sensitive"]
u, p, r = mwu(sens[sens.has_fragrance].rating, sens[~sens.has_fragrance].rating)
add("Sensitive skin: fragrance vs no fragrance rating (Mann-Whitney)", "yes", u, p, r, "rank-biserial")
oth = known[known.skin_type != "Sensitive"]
u, p, r = mwu(oth[oth.has_fragrance].rating, oth[~oth.has_fragrance].rating)
add("Other skin types: fragrance vs no fragrance (Mann-Whitney)", "no (null)", u, p, r, "rank-biserial")
rho, p = stats.spearmanr(df.price, df.rating)
add("Price vs rating (Spearman)", "no (null)", rho, p, rho, "rho")
u, p, r = mwu(df[df.rating <= 2].review_words, df[df.rating > 2].review_words)
add("Review length: negative vs other (Mann-Whitney)", "yes", u, p, r, "rank-biserial")
tab = pd.crosstab(df.category.eq("Sunscreen"), df.review_date.dt.month.between(5, 8))
chi2, p, v = cramers(tab)
add("Sunscreen share: May-Aug vs other months (chi-square)", "yes", chi2, p, v, "Cramer's V")
tab = pd.crosstab(df.review_date.dt.year, df.has_retinol)
chi2, p, v = cramers(tab)
add("Retinol product share across 2021-2023 (chi-square)", "yes", chi2, p, v, "Cramer's V")
M["retinol_by_year"] = df.groupby(df.review_date.dt.year).has_retinol.mean().round(3).to_dict()
M["sun_summer"] = {"summer": float(df[df.review_date.dt.month.between(5, 8)].category.eq("Sunscreen").mean()),
                   "other": float(df[~df.review_date.dt.month.between(5, 8)].category.eq("Sunscreen").mean())}
M["frag_sens"] = {"sens_frag": float(sens[sens.has_fragrance].rating.mean()), "sens_nofrag": float(sens[~sens.has_fragrance].rating.mean()),
                  "oth_frag": float(oth[oth.has_fragrance].rating.mean()), "oth_nofrag": float(oth[~oth.has_fragrance].rating.mean())}
M["words_neg"] = {"neg": float(df[df.rating <= 2].review_words.median()), "other": float(df[df.rating > 2].review_words.median())}

# ---------------- 3. feature engineering (history up to the cut-off only) ----------------
h = df[df.review_date <= CUT].sort_values(["user_id", "review_date"]).copy()
h["rel_rating"] = h.rating - h.groupby("product_id").rating.transform("mean")
h["premium"] = h.price / h.groupby("category").price.transform("median")
h["gap"] = h.groupby("user_id").review_date.diff().dt.days
h["is_sun"] = h.category.eq("Sunscreen"); h["summer"] = h.review_date.dt.month.between(5, 8); h["neg"] = h.rating <= 2
d90, d180 = CUT - pd.Timedelta(days=90), CUT - pd.Timedelta(days=180)
r90 = h[h.review_date > d90].groupby("product_id").size(); r180 = h[(h.review_date <= d90) & (h.review_date > d180)].groupby("product_id").size()
pids = h.product_id.unique()
mom = (r90.reindex(pids).fillna(0) + 1) / (r180.reindex(pids).fillna(0) + 1)
h["prod_momentum"] = h.product_id.map(mom)
g = h.groupby("user_id")
F = pd.DataFrame({"n_reviews": g.size(), "first": g.review_date.min(), "last": g.review_date.max(),
    "avg_gap": g.gap.mean(), "gap_std": g.gap.std(), "last_gap": g.gap.last(),
    "mean_rating": g.rating.mean(), "last_rating": g.rating.last(), "min_rating": g.rating.min(),
    "share_neg": g.neg.mean(), "rel_rating": g.rel_rating.mean(), "avg_words": g.review_words.mean(),
    "log_avg_price": np.log(g.price.mean()), "premium_ratio": g.premium.mean(),
    "retinol_share": g.has_retinol.mean(), "fragrance_share": g.has_fragrance.mean(), "niacinamide_share": g.has_niacinamide.mean(),
    "avg_n_ingr": g.n_ingredients.mean(), "sunscreen_share": g.is_sun.mean(), "summer_share": g.summer.mean(),
    "trend_follow": g.prod_momentum.mean(), "n_categories": g.category.nunique(), "skin_type": g.skin_type.first()})
F["gap_std"] = F.gap_std.fillna(0)
F["recency_days"] = (CUT - F["last"]).dt.days; F["tenure_days"] = (CUT - F["first"]).dt.days
F["gap_ratio"] = F.last_gap / F.avg_gap
u90 = h[h.review_date > d90].groupby("user_id").size().reindex(F.index).fillna(0)
u180 = h[(h.review_date <= d90) & (h.review_date > d180)].groupby("user_id").size().reindex(F.index).fillna(0)
F["momentum"] = (u90 + 1) / (u180 + 1)
cnt = h.groupby(["user_id", "category"]).size().unstack(fill_value=0); pr = cnt.div(cnt.sum(1), axis=0)
F["cat_entropy"] = -(pr * np.log(pr.where(pr > 0, 1))).sum(1); F["top_cat_share"] = pr.max(1); F["top_category"] = pr.idxmax(1)
F["rating_trend"] = g.apply(lambda x: np.polyfit(np.arange(len(x)), x.rating, 1)[0] if len(x) >= 3 else 0.0)
F["last_month_sin"] = np.sin(2 * np.pi * F["last"].dt.month / 12); F["last_month_cos"] = np.cos(2 * np.pi * F["last"].dt.month / 12)
F["sensitive_x_fragrance"] = (F.skin_type == "Sensitive").astype(float) * F.fragrance_share
F["gap_ratio_x_neg"] = F.gap_ratio * F.share_neg
F["recency_x_logfreq"] = F.recency_days * np.log1p(F.n_reviews)
future = df[(df.review_date > CUT) & (df.review_date <= END)].user_id.unique()
F["churn"] = (~F.index.isin(future)).astype(int)
F = F[F.n_reviews >= 2].copy()
M["modelling"] = {"users": len(F), "churn_rate": float(F.churn.mean()), "cutoff": str(CUT.date()), "window_end": str(END.date())}

BASE = ["recency_days", "n_reviews", "tenure_days"]
G_GAP = ["avg_gap", "gap_std", "last_gap", "gap_ratio", "momentum", "recency_x_logfreq"]
G_RATING = ["mean_rating", "last_rating", "min_rating", "share_neg", "rel_rating", "rating_trend", "avg_words", "gap_ratio_x_neg"]
G_PROFILE = ["n_categories", "cat_entropy", "top_cat_share", "log_avg_price", "premium_ratio", "retinol_share", "fragrance_share",
             "niacinamide_share", "avg_n_ingr", "sensitive_x_fragrance"]
G_TIME = ["sunscreen_share", "summer_share", "last_month_sin", "last_month_cos", "trend_follow"]
CATC = ["skin_type", "top_category"]
LOGCOLS = {"recency_days", "n_reviews", "tenure_days", "avg_gap", "gap_std", "last_gap", "momentum", "avg_words", "premium_ratio", "recency_x_logfreq"}
skews = {c: (float(stats.skew(F[c])), float(stats.skew(np.log1p(F[c])))) for c in ["n_reviews", "recency_days", "avg_gap", "avg_words"]}
M["skew_before_after"] = skews

def prep(num, cat):
    lg = [c for c in num if c in LOGCOLS]; rest = [c for c in num if c not in LOGCOLS]
    parts = []
    if lg: parts.append(("log", Pipeline([("i", SimpleImputer(strategy="median")), ("l", FunctionTransformer(np.log1p)), ("s", StandardScaler())]), lg))
    if rest: parts.append(("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), rest))
    if cat: parts.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat))
    return ColumnTransformer(parts)

cv = StratifiedKFold(5, shuffle=True, random_state=0)
def score(num, cat, kind):
    model = LogisticRegression(max_iter=3000) if kind == "lr" else HistGradientBoostingClassifier(random_state=0)
    pipe = Pipeline([("p", prep(num, cat)), ("m", model)])
    s = cross_val_score(pipe, F[num + cat], F.churn, cv=cv, scoring="roc_auc"); return float(s.mean()), float(s.std())
sets = {"Baseline (recency, frequency, tenure)": (BASE, []), "+ gap and momentum": (BASE + G_GAP, []),
        "+ rating and text": (BASE + G_RATING, []), "+ profile (category, price, ingredients, skin)": (BASE + G_PROFILE, CATC),
        "+ season and trend": (BASE + G_TIME, []), "All engineered features": (BASE + G_GAP + G_RATING + G_PROFILE + G_TIME, CATC)}
M["auc"] = {k: {"lr": score(n, c, "lr"), "gb": score(n, c, "gb")} for k, (n, c) in sets.items()}
allnum, allcat = sets["All engineered features"]
oof = cross_val_predict(Pipeline([("p", prep(allnum, allcat)), ("m", HistGradientBoostingClassifier(random_state=0))]),
                        F[allnum + allcat], F.churn, cv=cv, method="predict_proba")[:, 1]
F["risk"] = oof; top = F.sort_values("risk", ascending=False)
n10, n20 = int(len(F) * .10), int(len(F) * .20)
M["targeting"] = {"n_users": len(F), "base_rate": float(F.churn.mean()), "top10_n": n10, "top10_churn": float(top.churn.head(n10).mean()),
                  "lift10": float(top.churn.head(n10).mean() / F.churn.mean()), "top20_capture": float(top.churn.head(n20).sum() / F.churn.sum()),
                  "top10_capture": float(top.churn.head(n10).sum() / F.churn.sum())}

# ---------------- 4. user-level tests, multivariate EDA ----------------
F["any_neg"] = F.share_neg > 0
tab = pd.crosstab(F.any_neg, F.churn); chi2, p, v = cramers(tab)
add("Churn vs any negative rating before cut-off (chi-square)", "yes", chi2, p, v, "Cramer's V")
M["churn_by_neg"] = F.groupby("any_neg").churn.mean().round(3).to_dict()
u, p, r = mwu(F[F.churn == 1].avg_gap, F[F.churn == 0].avg_gap)
add("Average gap: churned vs retained (Mann-Whitney)", "no (reverse found)", u, p, r, "rank-biserial")
F["gap_q"] = pd.qcut(F.avg_gap, 4, labels=["Q1 shortest", "Q2", "Q3", "Q4 longest"])
M["churn_by_gapq"] = F.groupby("gap_q", observed=True).churn.mean().round(3).to_dict()
# Benjamini-Hochberg adjusted p
ps = np.array([t["p"] for t in tests]); order = np.argsort(ps); m = len(ps)
adj = np.empty(m); run = 1.0
for rank, i in enumerate(order[::-1]):
    k = m - rank; run = min(run, ps[i] * m / k); adj[i] = run
for t, a in zip(tests, adj): t["p_adj"] = float(a)
M["tests"] = tests

num_all = BASE + G_GAP + G_RATING + G_PROFILE + G_TIME
corr = F[num_all].corr(method="spearman")
pairs = [(a, b, float(corr.loc[a, b])) for i, a in enumerate(num_all) for b in num_all[i + 1:] if abs(corr.loc[a, b]) >= .8]
M["high_corr_pairs"] = sorted(pairs, key=lambda x: -abs(x[2]))[:8]
Z = prep(num_all, []).fit_transform(F[num_all])
pca = PCA(5, random_state=0).fit(Z); M["pca"] = [float(x) for x in pca.explained_variance_ratio_]
P2 = pca.transform(Z)[:, :2]
SEG = ["recency_days", "n_reviews", "avg_gap", "share_neg", "mean_rating", "cat_entropy", "log_avg_price"]
Zs = prep(SEG, []).fit_transform(F[SEG])
sil = {k: float(silhouette_score(Zs, KMeans(k, n_init=10, random_state=0).fit_predict(Zs))) for k in range(2, 7)}
k = max(range(3, 6), key=lambda x: sil[x]); F["segment"] = KMeans(k, n_init=10, random_state=0).fit_predict(Zs)
M["silhouette"] = sil; M["k"] = k
prof = F.groupby("segment").agg(users=("churn", "size"), churn=("churn", "mean"), recency=("recency_days", "median"), reviews=("n_reviews", "median"),
                                avg_gap=("avg_gap", "median"), share_neg=("share_neg", "mean"), mean_rating=("mean_rating", "mean"), entropy=("cat_entropy", "mean")).round(2)
M["segments"] = prof.reset_index().to_dict("records")

# ---------------- 5. figures ----------------
fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.2))
df.rating.value_counts().sort_index().plot.bar(ax=ax[0, 0], color="#1F4E79"); ax[0, 0].set_title("Rating counts"); ax[0, 0].set_xlabel("")
ax[0, 1].hist(np.log10(df.price), bins=40, color="#1F4E79"); ax[0, 1].set_title("Price (log10 scale)")
ax[1, 0].hist(per_user, bins=40, color="#1F4E79"); ax[1, 0].set_title("Reviews per user"); ax[1, 0].set_yscale("log")
ax[1, 1].hist(df.review_words, bins=40, color="#1F4E79"); ax[1, 1].set_title("Review length (words)")
fig.tight_layout(); fig.savefig("fig_uni.png", dpi=200); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
ms = df.groupby(df.review_date.dt.month).apply(lambda x: x.category.eq("Sunscreen").mean() * 100)
ax[0].plot(ms.index, ms.values, marker="o", color="#C0504D"); ax[0].set_title("Sunscreen share of reviews by month"); ax[0].set_xlabel("Month"); ax[0].set_ylabel("%")
q = df.groupby(df.review_date.dt.to_period("Q")).has_retinol.mean() * 100
ax[1].plot(range(len(q)), q.values, marker="o", color="#1F4E79"); ax[1].set_xticks(range(len(q))[::2], [str(x) for x in q.index][::2], rotation=45, fontsize=7)
ax[1].set_title("Reviews of retinol products, by quarter"); ax[1].set_ylabel("%")
fig.tight_layout(); fig.savefig("fig_time.png", dpi=200); plt.close()

hm = known.assign(fragrance=np.where(known.has_fragrance, "With fragrance", "No fragrance")).pivot_table(index="skin_type", columns="fragrance", values="rating", aggfunc="mean")
fig, ax = plt.subplots(figsize=(4.6, 3.0)); sns.heatmap(hm, annot=True, fmt=".2f", cmap="Blues", cbar=False, ax=ax); ax.set_xlabel(""); ax.set_ylabel("")
ax.set_title("Mean rating by skin type and fragrance"); fig.tight_layout(); fig.savefig("fig_heat.png", dpi=200); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
F.groupby("gap_q", observed=True).churn.mean().mul(100).plot.bar(ax=ax[0], color="#1F4E79"); ax[0].set_title("Churn % by average gap quartile"); ax[0].set_xlabel(""); ax[0].tick_params(axis="x", rotation=20)
F.groupby("any_neg").churn.mean().mul(100).rename({False: "No negative rating", True: "Had a negative rating"}).plot.bar(ax=ax[1], color="#C0504D")
ax[1].set_title("Churn % by past negative rating"); ax[1].set_xlabel(""); ax[1].tick_params(axis="x", rotation=10)
fig.tight_layout(); fig.savefig("fig_churn.png", dpi=200); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.2))
ax[0].scatter(P2[:, 0], P2[:, 1], c=np.where(F.churn == 1, "#C0504D", "#1F4E79"), s=6, alpha=.5)
ax[0].set_title("PCA of engineered features"); ax[0].set_xlabel(f"PC1 ({M['pca'][0]*100:.0f}%)"); ax[0].set_ylabel(f"PC2 ({M['pca'][1]*100:.0f}%)")
ax[0].scatter([], [], c="#C0504D", s=10, label="churned"); ax[0].scatter([], [], c="#1F4E79", s=10, label="retained"); ax[0].legend(fontsize=7, frameon=False)
prof["churn"].mul(100).plot.bar(ax=ax[1], color="#1F4E79"); ax[1].set_title(f"Churn % by K-means segment (k={k})"); ax[1].set_xlabel("Segment"); ax[1].tick_params(axis="x", rotation=0)
fig.tight_layout(); fig.savefig("fig_multi.png", dpi=200); plt.close()

names = list(M["auc"]); y = np.arange(len(names))
fig, ax = plt.subplots(figsize=(7.2, 3.5))
ax.barh(y - .2, [M["auc"][n]["lr"][0] for n in names], .38, xerr=[M["auc"][n]["lr"][1] for n in names], label="Logistic regression", color="#7F9FC5")
ax.barh(y + .2, [M["auc"][n]["gb"][0] for n in names], .38, xerr=[M["auc"][n]["gb"][1] for n in names], label="Gradient boosting", color="#1F4E79")
ax.set_yticks(y, names, fontsize=8); ax.invert_yaxis(); ax.set_xlim(.5, 1); ax.set_xlabel("Cross-validated ROC-AUC"); ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
fig.tight_layout(); fig.savefig("fig_auc.png", dpi=200); plt.close()

key = ["recency_days", "n_reviews", "avg_gap", "gap_ratio", "momentum", "mean_rating", "share_neg", "avg_words", "cat_entropy", "log_avg_price", "fragrance_share", "trend_follow"]
fig, ax = plt.subplots(figsize=(5.6, 4.6)); sns.heatmap(F[key].corr(method="spearman"), cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot=True, fmt=".1f", annot_kws={"size": 6}, ax=ax, cbar_kws={"shrink": .7})
ax.set_title("Spearman correlation of key features"); fig.tight_layout(); fig.savefig("fig_corr.png", dpi=200); plt.close()

# ---------------- 6. workflow flowchart ----------------
import graphviz
gz = graphviz.Digraph(format="png"); gz.attr(rankdir="TB", dpi="200", nodesep="0.25", ranksep="0.28")
gz.attr("node", shape="box", style="rounded,filled", fontname="Helvetica", fontsize="10", margin="0.12,0.06", width="4.8")
steps = [("1  Business questions", "Turn H1 to H4 into questions a brand could act on", "#F2F2F2"),
 ("2  Data audit", "Row counts, dates, types, missing values, cut-off date fixed", "#D9E2F3"),
 ("3  Univariate EDA", "Distributions, skew, outliers; decide transformations", "#D9E2F3"),
 ("4  Bivariate EDA and tests", "Group differences, correlations, effect sizes, adjusted p-values", "#D9E2F3"),
 ("5  Time and trend EDA", "Seasonality, trends, cohort and gap patterns", "#D9E2F3"),
 ("6  Multi-dimensional EDA", "Correlation matrix, PCA, K-means segments", "#FCE4D6"),
 ("7  Feature engineering", "Behaviour, rating, profile, season and trend features; interactions", "#FCE4D6"),
 ("8  Feature checks", "Leakage (past data only), skew, collinearity, missing values", "#FCE4D6"),
 ("9  Model check", "Do the features improve cross-validated accuracy?", "#E2EFDA"),
 ("10  Business insight", "Translate findings into actions and measure the value", "#E2EFDA")]
for i, (a, b, c) in enumerate(steps):
    gz.node(str(i), f"<<B>{a}</B><BR/><FONT POINT-SIZE='9'>{b}</FONT>>", fillcolor=c, color="#7F7F7F")
    if i: gz.edge(str(i - 1), str(i), color="#7F7F7F")
gz.edge("9", "0", constraint="false", style="dashed", color="#7F7F7F", label=" iterate", fontsize="9")
gz.render("eda_flow", cleanup=True)
F.drop(columns=["gap_q"]).to_csv("user_features.csv")
json.dump(M, open("metrics.json", "w"), indent=1, default=str)
print(json.dumps({k: M[k] for k in ["overview", "modelling", "pca", "silhouette", "k", "targeting", "churn_by_neg", "churn_by_gapq", "sun_summer", "frag_sens", "words_neg", "retinol_by_year", "skew_before_after", "high_corr_pairs"]}, indent=1, default=str))
for t in tests: print(f"{t['test'][:62]:62s} planted={t['planted']:9s} p={t['p']:.2e} adj={t['p_adj']:.2e} {t['effect_name']}={t['effect']:.3f}")
for n, v in M["auc"].items(): print(f"{n:50s} LR {v['lr'][0]:.3f}±{v['lr'][1]:.3f}  GB {v['gb'][0]:.3f}±{v['gb'][1]:.3f}")
print(prof.to_string())
