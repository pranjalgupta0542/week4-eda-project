"""Simulated, already-cleaned beauty review data WITH KNOWN PATTERNS (so we can test whether EDA finds them).
Planted: (1) sunscreen reviews peak in May-Aug; (2) retinol products grow over time; (3) Sensitive skin rates
fragrance products lower; (4) negative reviews are longer; (5) a negative rating raises the chance a user stops;
(6) price has NO effect on rating (a null result on purpose). Everything is synthetic."""
import numpy as np, pandas as pd
rng = np.random.default_rng(7)
START, END = pd.Timestamp("2021-01-01"), pd.Timestamp("2023-12-31")
CATS = ["Moisturizer", "Serum", "Cleanser", "Sunscreen", "Mask", "Toner"]
BRANDS = ["Glow Lab", "Pure Roots", "Dermis Co", "Velvet Skin", "Aqua Bloom", "Nature and Co", "Lumi Care",
          "Sun Guard", "Calm Botanics", "Clear Theory", "Rose Atelier", "Hydra Lab"]
SKIN = ["Dry", "Oily", "Combination", "Normal", "Sensitive"]
NP, NU = 300, 1500
cat = rng.choice(CATS, NP)
prod = pd.DataFrame({"product_id": [f"P{1000+i}" for i in range(NP)], "category": cat, "brand": rng.choice(BRANDS, NP),
    "price": np.round(np.exp(rng.normal(3.3, .55, NP)), 2),
    "has_retinol": rng.random(NP) < np.where(np.isin(cat, ["Serum", "Moisturizer"]), .40, .05),
    "has_fragrance": rng.random(NP) < .40, "has_niacinamide": rng.random(NP) < .30,
    "n_ingredients": rng.integers(8, 26, NP)})
users = pd.DataFrame({"user_id": [f"U{i:04d}" for i in range(1, NU + 1)],
    "skin_type": rng.choice(SKIN + ["Unknown"], NU, p=[.2, .2, .2, .15, .10, .15]),
    "mean_gap": np.exp(rng.normal(np.log(55), .5, NU)), "fav": rng.choice(CATS, NU),
    "join": START + pd.to_timedelta(rng.integers(0, 730, NU), unit="D")})
by_cat = {c: prod[prod.category == c].reset_index(drop=True) for c in CATS}
rows = []
for u in users.itertuples():
    d = u.join
    while d <= END:
        w = np.array([2.5 if c == u.fav else 1.0 for c in CATS]) * np.array([ (3.0 if 5 <= d.month <= 8 else .6) if c == "Sunscreen" else 1.0 for c in CATS])
        c = rng.choice(CATS, p=w / w.sum()); pc = by_cat[c]
        t = (d - START).days / (END - START).days
        pw = 1 + pc.has_retinol.to_numpy() * 1.5 * t * 2
        p = pc.iloc[rng.choice(len(pc), p=pw / pw.sum())]
        lat = 4.3 - (0.9 if (u.skin_type == "Sensitive" and p.has_fragrance) else 0) + rng.normal(0, .9)
        r = int(np.clip(round(lat), 1, 5))
        words = int(rng.poisson(18) + (14 if r <= 2 else 0))
        rows.append((u.user_id, p.product_id, d.normalize(), r, words))
        if rng.random() < .04 + .22 * (r <= 2): break
        d = d + pd.to_timedelta(max(3, rng.exponential(u.mean_gap)), unit="D")
df = pd.DataFrame(rows, columns=["user_id", "product_id", "review_date", "rating", "review_words"])
df = df.merge(prod, on="product_id").merge(users[["user_id", "skin_type"]], on="user_id")
df.to_csv("eda_reviews.csv", index=False)
print(len(df), "reviews |", df.user_id.nunique(), "users |", df.review_date.min().date(), "to", df.review_date.max().date())
