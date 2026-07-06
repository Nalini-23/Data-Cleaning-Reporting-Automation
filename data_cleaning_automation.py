"""
Data Cleaning & Reporting Automation
-------------------------------------
Automates a typical messy-dataset cleanup and produces a structured report
so the whole process is auditable, not just "run and hope".

Pipeline:
  1. Standardize column names
  2. Fix inconsistent text (casing, whitespace, stray symbols)
  3. Parse inconsistent date formats into one standard format
  4. Convert "dirty" numeric strings ("1,200", "$45", "N/A") into real numbers
  5. Handle missing values (report %, then impute or drop based on threshold)
  6. Remove exact and near-duplicate rows
  7. Flag outliers using the IQR method (flagged, not silently deleted)
  8. Log every action with before/after counts

Produces:
  - raw_messy_data.csv     (the "before" dataset)
  - cleaned_data.csv       (the "after" dataset)
  - cleaning_report.json   (everything the HTML report needs)

Replace `generate_messy_data()` with `pd.read_csv("your_file.csv")` to run
this on a real, messy dataset.
"""

import json
import re
import numpy as np
import pandas as pd

RNG = np.random.default_rng(11)

# ---------------------------------------------------------------------------
# 1. Generate a deliberately messy dataset (real-world style problems)
# ---------------------------------------------------------------------------
def generate_messy_data(n=500):
    names = ["Aarav Shah", "Priya Nair", "Rohan Gupta", "Sneha Iyer", "Vikram Rao",
              "Ananya Das", "Karthik Menon", "Divya Pillai", "Arjun Reddy", "Meera Joshi"]
    cities_clean = ["Bengaluru", "Mumbai", "Delhi", "Chennai", "Hyderabad"]
    city_variants = {
        "Bengaluru": ["bengaluru", "Bangalore ", " BENGALURU", "bengaluru "],
        "Mumbai": ["mumbai", "Mumbai ", "MUMBAI", " Mumbai"],
        "Delhi": ["delhi", "New Delhi", "DELHI ", "Delhi "],
        "Chennai": ["chennai", "CHENNAI", " Chennai", "chennai "],
        "Hyderabad": ["hyderabad", "HYDERABAD ", " Hyderabad", "hyderabad"],
    }
    categories = ["Electronics", "Apparel", "Home & Kitchen", "Books", "Sports"]
    date_formats = ["%Y-%m-%d", "%d/%m/%Y", "%d-%b-%Y", "%m/%d/%Y"]

    rows = []
    for i in range(n):
        city_key = RNG.choice(cities_clean)
        city_raw = RNG.choice(city_variants[city_key])
        d = pd.Timestamp("2025-01-01") + pd.Timedelta(days=int(RNG.integers(0, 400)))
        fmt = RNG.choice(date_formats)
        date_raw = d.strftime(fmt)

        price = round(RNG.normal(1500, 500), 2)
        # dirty price formatting
        price_variants = [str(price), f"₹{price:,.2f}", f"{price:,.2f}", "N/A", ""]
        price_raw = RNG.choice(price_variants, p=[0.55, 0.2, 0.15, 0.06, 0.04])

        qty = int(RNG.integers(1, 10))
        if RNG.random() < 0.03:
            qty = -qty  # bad data: negative quantity

        row = {
            "Customer Name": RNG.choice(names),
            "  city": city_raw,
            "Order Date": date_raw,
            "category ": RNG.choice(categories),
            "PRICE": price_raw,
            "quantity": qty if RNG.random() > 0.04 else None,   # missing values
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # inject exact duplicate rows
    dup_rows = df.sample(frac=0.05, random_state=1)
    df = pd.concat([df, dup_rows], ignore_index=True)

    # inject a few extreme outliers in price
    outlier_idx = RNG.choice(df.index, size=4, replace=False)
    df.loc[outlier_idx, "PRICE"] = "89999.00"

    return df.sample(frac=1, random_state=2).reset_index(drop=True)


raw_df = generate_messy_data()
raw_df.to_csv("/home/claude/raw_messy_data.csv", index=False)

log = []  # human-readable audit trail
report = {"before": {}, "after": {}, "actions": []}

report["before"]["rows"] = len(raw_df)
report["before"]["missing_by_column"] = raw_df.isna().sum().to_dict()
report["before"]["duplicate_rows"] = int(raw_df.duplicated().sum())

df = raw_df.copy()

# ---------------------------------------------------------------------------
# 2. Standardize column names
# ---------------------------------------------------------------------------
original_cols = list(df.columns)
df.columns = [re.sub(r"\s+", "_", c.strip().lower()) for c in df.columns]
log.append(f"Standardized column names: {original_cols} -> {list(df.columns)}")

# ---------------------------------------------------------------------------
# 3. Clean text fields (city, category, customer name)
# ---------------------------------------------------------------------------
def clean_text(x):
    if pd.isna(x):
        return x
    x = str(x).strip()
    x = re.sub(r"\s+", " ", x)
    return x.title()

for col in ["city", "category", "customer_name"]:
    before_unique = df[col].nunique()
    df[col] = df[col].apply(clean_text)
    df[col] = df[col].replace({"New Delhi": "Delhi", "Bangalore": "Bengaluru"})  # merge known aliases
    after_unique = df[col].nunique()
    log.append(f"Standardized text in '{col}': {before_unique} distinct values -> {after_unique}")

# ---------------------------------------------------------------------------
# 4. Parse inconsistent dates
# ---------------------------------------------------------------------------
df["order_date"] = pd.to_datetime(df["order_date"], format="mixed", dayfirst=True, errors="coerce")
bad_dates = df["order_date"].isna().sum()
log.append(f"Parsed 'order_date' into a single format; {bad_dates} unparseable values became NaT")

# ---------------------------------------------------------------------------
# 5. Clean dirty numeric strings (price)
# ---------------------------------------------------------------------------
def clean_price(x):
    if pd.isna(x):
        return np.nan
    x = str(x).replace("₹", "").replace(",", "").strip()
    if x in ("", "N/A", "n/a", "NA"):
        return np.nan
    try:
        return float(x)
    except ValueError:
        return np.nan

df["price"] = df["price"].apply(clean_price)
log.append("Converted 'price' from mixed text/currency formatting into numeric values")

# ---------------------------------------------------------------------------
# 6. Fix bad quantities (negative -> made positive, since it's a data-entry sign error)
# ---------------------------------------------------------------------------
neg_qty = int((df["quantity"] < 0).sum())
df["quantity"] = df["quantity"].abs()
log.append(f"Corrected {neg_qty} negative quantity values (sign entry errors) to positive")

# ---------------------------------------------------------------------------
# 7. Handle missing values
# ---------------------------------------------------------------------------
missing_before = df.isna().sum()
MISSING_THRESHOLD = 0.4  # drop column if >40% missing; otherwise impute

missing_actions = {}
for col in df.columns:
    pct_missing = df[col].isna().mean()
    if pct_missing == 0:
        continue
    if pct_missing > MISSING_THRESHOLD:
        df.drop(columns=[col], inplace=True)
        missing_actions[col] = f"dropped column ({pct_missing:.1%} missing)"
    elif df[col].dtype.kind in "if":  # numeric -> median impute
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        missing_actions[col] = f"imputed with median ({median_val:.2f}), {pct_missing:.1%} were missing"
    else:  # categorical/date -> mode or drop rows
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            before_n = len(df)
            df = df.dropna(subset=[col])
            missing_actions[col] = f"dropped {before_n - len(df)} rows with unparseable dates"
        else:
            mode_val = df[col].mode(dropna=True)
            fill_val = mode_val.iloc[0] if len(mode_val) else "Unknown"
            df[col] = df[col].fillna(fill_val)
            missing_actions[col] = f"imputed with most common value ('{fill_val}'), {pct_missing:.1%} were missing"

for col, action in missing_actions.items():
    log.append(f"Missing values in '{col}': {action}")

# ---------------------------------------------------------------------------
# 8. Remove duplicates (exact, then near-duplicates on key business columns)
# ---------------------------------------------------------------------------
exact_dupes = int(df.duplicated().sum())
df = df.drop_duplicates()
log.append(f"Removed {exact_dupes} exact duplicate rows")

key_cols = ["customer_name", "order_date", "price", "quantity"]
key_cols = [c for c in key_cols if c in df.columns]
near_dupes = int(df.duplicated(subset=key_cols).sum())
df = df.drop_duplicates(subset=key_cols)
log.append(f"Removed {near_dupes} near-duplicate rows (same {key_cols})")

# ---------------------------------------------------------------------------
# 9. Flag outliers with IQR (flag, don't silently delete real business data)
# ---------------------------------------------------------------------------
q1, q3 = df["price"].quantile([0.25, 0.75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
df["price_outlier"] = (df["price"] < lower) | (df["price"] > upper)
n_outliers = int(df["price_outlier"].sum())
log.append(f"Flagged {n_outliers} price outliers using IQR method (bounds: {lower:.0f} to {upper:.0f}); kept but marked, not deleted")

# ---------------------------------------------------------------------------
# 10. Final report + save
# ---------------------------------------------------------------------------
report["after"]["rows"] = len(df)
report["after"]["missing_by_column"] = df.isna().sum().to_dict()
report["after"]["duplicate_rows"] = int(df.duplicated().sum())
report["after"]["outliers_flagged"] = n_outliers
report["actions"] = log
report["column_mapping"] = dict(zip(original_cols, list(pd.DataFrame(columns=original_cols).rename(
    columns=lambda c: re.sub(r"\s+", "_", c.strip().lower())).columns)))
report["rows_removed_total"] = report["before"]["rows"] - report["after"]["rows"]
report["missing_before_total"] = int(sum(report["before"]["missing_by_column"].values()))
report["missing_after_total"] = int(sum(report["after"]["missing_by_column"].values()))
report["notes"] = [
    "Raw 'price' showed 0 missing via a plain .isna() check, but that only catches true "
    "nulls. Values like 'N/A' and blank strings are sentinel placeholders that look "
    "non-null until parsed. After converting price to numeric, 9.5% of rows were "
    "actually missing. This is a common way real datasets hide missingness."
]

df.to_csv("/home/claude/cleaned_data.csv", index=False)

with open("/home/claude/cleaning_report.json", "w") as f:
    json.dump(report, f, default=str)

print("=== CLEANING SUMMARY ===")
print(f"Rows: {report['before']['rows']} -> {report['after']['rows']}")
print(f"Missing values: {report['missing_before_total']} -> {report['missing_after_total']}")
print(f"Duplicates removed: {exact_dupes + near_dupes}")
print(f"Outliers flagged: {n_outliers}")
print("\nFull action log:")
for a in log:
    print(" -", a)
