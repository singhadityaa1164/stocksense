"""
Inventory logic for StockSense.

All numbers the chatbot reports come from these functions, never from the
language model's own memory. The model only decides WHICH function to call
and how to phrase the result.
"""

import difflib
import math
import random
import re
import threading
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

_HERE = Path(__file__).parent
# Works whether inventory.csv sits in a data/ folder or next to this file.
DATA_PATH = next((p for p in (_HERE / "data" / "inventory.csv", _HERE / "inventory.csv") if p.exists()),
                 _HERE / "inventory.csv")

REQUIRED_COLUMNS = [
    "sku", "product", "brand", "category", "unit_price_inr", "stock_on_hand",
    "reorder_level", "lead_time_days", "units_sold_last_30d", "supplier",
]
NUMERIC_COLUMNS = ["unit_price_inr", "stock_on_hand", "reorder_level",
                   "lead_time_days", "units_sold_last_30d"]

SAFETY_DAYS = 7      # buffer stock kept on top of lead-time demand
COVER_DAYS = 30      # a reorder should cover roughly one month of sales

# Per-session state. Streamlit runs each browser session's script in its own
# thread, so thread-local storage keeps one user's uploaded inventory and
# tickets from leaking into another user's session.
_state = threading.local()


# --------------------------------------------------------------------------
# Loading and validation
# --------------------------------------------------------------------------
def validate_inventory(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Check an inventory table. Returns (clean_df, problems)."""
    problems: list[str] = []
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return df, [f"Missing column(s): {', '.join(missing)}"]

    df = df[REQUIRED_COLUMNS]
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    bad = df[df[NUMERIC_COLUMNS].isna().any(axis=1)]
    if len(bad):
        problems.append(f"{len(bad)} row(s) had non-numeric values and were dropped: "
                        + ", ".join(bad["sku"].astype(str).head(5)))
        df = df.drop(bad.index)
    neg = df[(df[NUMERIC_COLUMNS] < 0).any(axis=1)]
    if len(neg):
        problems.append(f"{len(neg)} row(s) had negative numbers and were dropped: "
                        + ", ".join(neg["sku"].astype(str).head(5)))
        df = df.drop(neg.index)
    dupes = df[df["sku"].duplicated()]
    if len(dupes):
        problems.append(f"Duplicate SKU(s) removed: {', '.join(dupes['sku'].astype(str).head(5))}")
        df = df.drop_duplicates("sku")
    for col in NUMERIC_COLUMNS:
        df[col] = df[col].astype(int) if col != "unit_price_inr" else df[col].astype(float)
    df["lead_time_days"] = df["lead_time_days"].clip(lower=1)
    if df.empty:
        problems.append("No valid rows left after cleaning.")
    return df.reset_index(drop=True), problems


def load_default() -> pd.DataFrame:
    df, _ = validate_inventory(pd.read_csv(DATA_PATH))
    return df


def set_session(df: pd.DataFrame, tickets: list) -> None:
    """Bind this thread to the current user's inventory table and ticket list."""
    _state.inventory = df
    _state.tickets = tickets


def get_inventory() -> pd.DataFrame:
    if getattr(_state, "inventory", None) is None:
        _state.inventory = load_default()
    return _state.inventory


def _tickets() -> list:
    if getattr(_state, "tickets", None) is None:
        _state.tickets = []
    return _state.tickets


# --------------------------------------------------------------------------
# Core calculations
# --------------------------------------------------------------------------
def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add sales-velocity metrics used by alerts and reorder suggestions."""
    out = df.copy()
    out["daily_sales"] = (out["units_sold_last_30d"] / 30).round(2)
    out["days_of_cover"] = out.apply(
        lambda r: round(r["stock_on_hand"] / r["daily_sales"], 1) if r["daily_sales"] > 0 else math.inf,
        axis=1,
    )
    out["safety_stock"] = (out["daily_sales"] * SAFETY_DAYS).apply(math.ceil)
    out["reorder_point"] = (out["daily_sales"] * out["lead_time_days"]).apply(math.ceil) + out["safety_stock"]
    target = (out["daily_sales"] * (out["lead_time_days"] + COVER_DAYS)).apply(math.ceil) + out["safety_stock"]
    out["suggested_order_qty"] = (target - out["stock_on_hand"]).clip(lower=0).astype(int)

    def status(r):
        if r["stock_on_hand"] == 0:
            return "OUT OF STOCK"
        if r["stock_on_hand"] <= r["reorder_level"] or r["days_of_cover"] < r["lead_time_days"]:
            return "LOW"
        if r["days_of_cover"] > 90:
            return "OVERSTOCK"
        return "OK"

    out["status"] = out.apply(status, axis=1)
    return out


def _row_to_dict(r: pd.Series) -> dict:
    cover = r["days_of_cover"]
    return {
        "sku": r["sku"],
        "product": r["product"],
        "brand": r["brand"],
        "category": r["category"],
        "price_inr": int(r["unit_price_inr"]),
        "stock_on_hand": int(r["stock_on_hand"]),
        "reorder_level": int(r["reorder_level"]),
        "units_sold_last_30d": int(r["units_sold_last_30d"]),
        "days_of_cover": "no recent sales" if cover == math.inf else float(cover),
        "lead_time_days": int(r["lead_time_days"]),
        "status": r["status"],
    }


# --------------------------------------------------------------------------
# Fuzzy product search
# --------------------------------------------------------------------------
_STOP = {"do", "we", "have", "any", "how", "many", "is", "are", "the", "a", "an", "of",
         "in", "stock", "units", "left", "there", "what", "about", "for", "me", "show",
         "check", "available", "availability", "please", "pls", "you", "got", "still",
         "status", "much", "can", "i", "tell", "price", "and", "with", "our", "us", "to",
         "inch", "inches", "model", "models", "need", "want", "looking", "find", "right", "now", "today"}

_SYNONYMS = {
    "tv": "televisions", "tvs": "televisions", "television": "televisions",
    "phone": "mobiles", "phones": "mobiles", "mobile": "mobiles", "smartphone": "mobiles",
    "smartphones": "mobiles", "laptop": "laptops", "notebook": "laptops",
    "ac": "air conditioners", "acs": "air conditioners", "aircon": "air conditioners",
    "fridge": "refrigerators", "fridges": "refrigerators", "refrigerator": "refrigerators",
    "washer": "washing machines", "washing": "washing machines",
    "earbuds": "earbuds", "headphone": "headphones", "speaker": "speaker",
    "watch": "watch", "smartwatch": "smartwatch", "mac": "macbook",
    "samsumg": "samsung", "iphon": "iphone", "aiphone": "iphone",
}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _match_score(query_tokens: list[str], haystack: str) -> float:
    hay_tokens = _tokens(haystack)
    if not query_tokens or not hay_tokens:
        return 0.0
    hits = 0.0
    word_hit = False
    for q in query_tokens:
        if q in hay_tokens or (len(q) > 3 and q in haystack.lower()):
            hits += 1
            word_hit = word_hit or not q.isdigit()
        elif not q.isdigit() and difflib.get_close_matches(q, hay_tokens, n=1, cutoff=0.8):
            hits += 0.7   # typo tolerance, e.g. "samsng"
            word_hit = True
    # A bare number ("5", "15") must not match on its own: "playstation 5" used to
    # return every product with a 5 in its name. Require at least one word to match
    # whenever the query contains words.
    if not word_hit and any(not q.isdigit() for q in query_tokens):
        return 0.0
    return hits / len(query_tokens)


def search_products(query: str, category: str = "", brand: str = "", max_price_inr: float = 0) -> dict:
    """Look up products in store inventory by name, model, brand or category, with typo tolerance.

    Args:
        query: What the user is looking for, e.g. "iphone 15", "55 inch samsung tv", "air fryer". Use "" to list everything matching the filters.
        category: Optional category filter, e.g. "Televisions", "Mobiles", "Laptops", "Air Conditioners", "Refrigerators", "Washing Machines", "Audio", "Accessories", "Kitchen Appliances", "Wearables".
        brand: Optional brand filter, e.g. "Samsung".
        max_price_inr: Optional maximum unit price in rupees. 0 means no limit.

    Returns:
        A dict with the matching products and their live stock, or a not-found message with close suggestions.
    """
    df = enrich(get_inventory())
    if category:
        cat_q = _SYNONYMS.get(category.lower().strip(), category.lower().strip())
        df = df[df["category"].str.lower().str.contains(cat_q.rstrip("s"), regex=False)]
    if brand:
        df = df[df["brand"].str.lower() == brand.lower().strip()]
    if max_price_inr and max_price_inr > 0:
        df = df[df["unit_price_inr"] <= max_price_inr]

    q_tokens = [_SYNONYMS.get(t, t) for t in _tokens(query or "") if t not in _STOP]
    q_tokens = [w for t in q_tokens for w in t.split()]
    if not q_tokens:
        matches = df
    else:
        hay = df["sku"] + " " + df["product"] + " " + df["brand"] + " " + df["category"]
        scores = hay.apply(lambda h: _match_score(q_tokens, h))
        best = scores.max() if len(scores) else 0
        if best < 0.5:
            suggestions = difflib.get_close_matches(
                (query or "").lower(), get_inventory()["product"].str.lower().tolist(), n=3, cutoff=0.5)
            return {"found": False,
                    "message": f"No product in inventory matches '{query}'.",
                    "did_you_mean": suggestions}
        # keep only the strongest matches so "iphone 15" does not also return "iphone 16"
        matches = df[scores >= max(0.5, best - 0.01)]

    if matches.empty:
        return {"found": False, "message": "No products match those filters.", "did_you_mean": []}
    matches = matches.head(12)
    return {"found": True, "count": len(matches),
            "products": [_row_to_dict(r) for _, r in matches.iterrows()]}


def get_low_stock_alerts(category: str = "") -> dict:
    """List products that are out of stock or running low (at/below reorder level, or will run out before new stock can arrive).

    Args:
        category: Optional category filter. Use "" for all categories.

    Returns:
        A dict with alert items sorted by urgency.
    """
    df = enrich(get_inventory())
    if category:
        cat_q = _SYNONYMS.get(category.lower().strip(), category.lower().strip())
        df = df[df["category"].str.lower().str.contains(cat_q.rstrip("s"), regex=False)]
    alerts = df[df["status"].isin(["OUT OF STOCK", "LOW"])].copy()
    alerts["_rank"] = alerts["status"].map({"OUT OF STOCK": 0, "LOW": 1})
    alerts = alerts.sort_values(["_rank", "days_of_cover"])
    return {
        "rule": ("LOW = stock at or below reorder level, OR days of cover (stock / daily sales) "
                 "shorter than supplier lead time."),
        "alert_count": len(alerts),
        "items": [_row_to_dict(r) for _, r in alerts.iterrows()],
    }


def get_reorder_suggestions(category: str = "", sku: str = "") -> dict:
    """Suggest how many units to reorder, based on last-30-day sales velocity, supplier lead time and a 7-day safety buffer.

    Args:
        category: Optional category filter. Use "" for all.
        sku: Optional single SKU code to calculate for. Use "" for all.

    Returns:
        A dict of suggested order quantities with the working shown.
    """
    df = enrich(get_inventory())
    if sku:
        df = df[df["sku"].str.upper() == sku.upper().strip()]
        if df.empty:
            return {"found": False, "message": f"SKU '{sku}' not found."}
    if category:
        cat_q = _SYNONYMS.get(category.lower().strip(), category.lower().strip())
        df = df[df["category"].str.lower().str.contains(cat_q.rstrip("s"), regex=False)]
    need = df[(df["stock_on_hand"] <= df["reorder_point"]) & (df["suggested_order_qty"] > 0)]
    if sku and need.empty:
        r = df.iloc[0]
        return {"found": True, "items": [], "message":
                f"{r['product']} does not need a reorder now: {int(r['stock_on_hand'])} in stock vs. "
                f"reorder point {int(r['reorder_point'])}."}
    need = need.sort_values("days_of_cover")
    items = []
    for _, r in need.iterrows():
        items.append({
            "sku": r["sku"], "product": r["product"], "supplier": r["supplier"],
            "stock_on_hand": int(r["stock_on_hand"]),
            "daily_sales": float(r["daily_sales"]),
            "lead_time_days": int(r["lead_time_days"]),
            "reorder_point": int(r["reorder_point"]),
            "suggested_order_qty": int(r["suggested_order_qty"]),
            "estimated_cost_inr": int(r["suggested_order_qty"] * r["unit_price_inr"] * 0.85),
            "working": (f"daily sales {r['daily_sales']} x ({int(r['lead_time_days'])} lead days + {COVER_DAYS} cover days)"
                        f" + {int(r['safety_stock'])} safety stock - {int(r['stock_on_hand'])} on hand"),
        })
    return {
        "method": (f"Suggested qty = daily sales x (lead time + {COVER_DAYS} days) + {SAFETY_DAYS}-day safety stock "
                   "- stock on hand. Cost assumes ~15% below retail price (approximate dealer margin)."),
        "count": len(items),
        "total_estimated_cost_inr": sum(i["estimated_cost_inr"] for i in items),
        "items": items,
        "note": "These are suggestions only. A store manager must approve before a purchase order is placed.",
    }


def get_inventory_summary() -> dict:
    """Give an overview of the whole store inventory: totals and status counts by category, plus slow-moving (overstock) items.

    Returns:
        A dict with per-category stock, value and alert counts.
    """
    df = enrich(get_inventory())
    df["stock_value"] = df["stock_on_hand"] * df["unit_price_inr"]
    by_cat = []
    for cat, g in df.groupby("category"):
        by_cat.append({
            "category": cat, "skus": len(g), "units_in_stock": int(g["stock_on_hand"].sum()),
            "stock_value_inr": int(g["stock_value"].sum()),
            "out_of_stock": int((g["status"] == "OUT OF STOCK").sum()),
            "low": int((g["status"] == "LOW").sum()),
        })
    over = df[df["status"] == "OVERSTOCK"].sort_values("days_of_cover", ascending=False)
    return {
        "total_skus": len(df),
        "total_units": int(df["stock_on_hand"].sum()),
        "total_stock_value_inr": int(df["stock_value"].sum()),
        "status_counts": {k: int(v) for k, v in df["status"].value_counts().items()},
        "by_category": by_cat,
        "slow_moving_overstock": [_row_to_dict(r) for _, r in over.iterrows()],
    }


def escalate_to_manager(reason: str, sku: str = "") -> dict:
    """Raise a ticket for the store manager. Use when the user asks for a human, wants something the bot cannot do (place an order, change a price, correct stock data), or the data looks wrong.

    Args:
        reason: Short description of what needs human attention.
        sku: Optional related SKU.

    Returns:
        The ticket ID and expected response time.
    """
    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    ticket_id = f"INV-{now:%y%m%d}-{random.randint(1000, 9999)}"
    _tickets().append({"ticket_id": ticket_id, "reason": reason[:300], "sku": sku,
                    "created": now.strftime("%d %b %Y %H:%M IST")})
    return {"ticket_id": ticket_id, "assigned_to": "Store Manager",
            "expected_response": "within 4 working hours", "status": "OPEN"}


TOOLS = [search_products, get_low_stock_alerts, get_reorder_suggestions,
         get_inventory_summary, escalate_to_manager]
