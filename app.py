"""
StockSense - AI inventory & stock query assistant for a small electronics retailer.
Run locally:  streamlit run app.py
"""
from __future__ import annotations

import io
import os
import time

import pandas as pd
import streamlit as st

import bot_engine as engine
import inventory_tools as inv

st.set_page_config(page_title="StockSense - Inventory Assistant", page_icon="📦", layout="wide")

MAX_MESSAGES_PER_SESSION = 40

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1100px;}
.ai-badge {display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.8rem;
           background:#e8f0fe; color:#1a4fb4; margin-right:6px;}
.basic-badge {display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.8rem;
              background:#fff4e5; color:#8a4b00; margin-right:6px;}
.small-note {font-size:0.8rem; color:#6b7280;}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
def init_state():
    ss = st.session_state
    ss.setdefault("messages", [])          # [{"role", "content", "tools"}]
    ss.setdefault("inventory", inv.load_default())
    ss.setdefault("data_source", "Sample data (40 SKUs)")
    ss.setdefault("tickets", [])
    ss.setdefault("bot", None)
    ss.setdefault("ai_error", None)
    ss.setdefault("last_submit", ("", 0.0))
    ss.setdefault("pending", None)


init_state()
inv.set_session(st.session_state.inventory, st.session_state.tickets)


def get_api_key() -> str | None:
    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except Exception:  # no secrets file
        key = None
    return key or os.environ.get("GEMINI_API_KEY") or st.session_state.get("user_key") or None


def get_bot():
    if st.session_state.bot is None:
        key = get_api_key()
        if key:
            try:
                st.session_state.bot = engine.GeminiBot(key)
            except Exception as e:  # noqa: BLE001
                st.session_state.ai_error = f"Could not start Gemini client: {e}"
    return st.session_state.bot


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📦 StockSense")
    page = st.radio("View", ["💬 Chat assistant", "📊 Inventory dashboard"], label_visibility="collapsed")

    st.divider()
    bot = get_bot()
    if bot and not st.session_state.ai_error:
        st.markdown(f"<span class='ai-badge'>AI mode</span> Gemini "
                    f"<code>{bot.model or bot.models[0]}</code>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='basic-badge'>Basic mode</span> keyword matching, no AI",
                    unsafe_allow_html=True)
        if st.session_state.ai_error:
            st.caption(f"⚠️ {st.session_state.ai_error}")
        if not get_api_key():
            k = st.text_input("Gemini API key (optional)", type="password",
                              help="Free key from aistudio.google.com. Used only for this browser session.")
            if k:
                st.session_state.user_key = k.strip()
                st.session_state.bot = None
                st.session_state.ai_error = None
                st.rerun()
    if st.session_state.ai_error and st.button("Retry AI mode"):
        st.session_state.bot = None
        st.session_state.ai_error = None
        st.rerun()

    st.divider()
    st.markdown("**Inventory data**")
    st.caption(f"Using: {st.session_state.data_source}")
    up = st.file_uploader("Upload your own CSV", type=["csv"], label_visibility="collapsed")
    if up is not None and st.session_state.get("uploaded_name") != up.name:
        try:
            raw = pd.read_csv(up)
            df, problems = inv.validate_inventory(raw)
            if problems and (df.empty or any(p.startswith("Missing") for p in problems)):
                st.error(" ".join(problems))
            else:
                st.session_state.inventory = df
                st.session_state.data_source = f"Uploaded: {up.name} ({len(df)} SKUs)"
                st.session_state.uploaded_name = up.name
                for p in problems:
                    st.warning(p)
                st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read that file: {e}")
    c1, c2 = st.columns(2)
    c1.download_button("CSV template", inv.load_default().head(3).to_csv(index=False),
                       "inventory_template.csv", "text/csv", width="stretch")
    if c2.button("Reset data", width="stretch"):
        st.session_state.inventory = inv.load_default()
        st.session_state.data_source = "Sample data (40 SKUs)"
        st.session_state.uploaded_name = None
        st.rerun()

    st.divider()
    if st.button("🗑️ Clear conversation", width="stretch"):
        st.session_state.messages = []
        if st.session_state.bot:
            st.session_state.bot.chat = None
        st.rerun()

    if st.session_state.tickets:
        st.markdown("**Manager tickets (this session)**")
        for t in st.session_state.tickets:
            st.caption(f"🎫 {t['ticket_id']} · {t['created']}\n\n{t['reason'][:80]}")

    st.divider()
    st.markdown(
        "<div class='small-note'>🤖 You are chatting with an <b>AI assistant</b>, not a person. "
        "In AI mode, your messages and the matching inventory rows are sent to Google's Gemini API "
        "for processing. Do not enter customer personal data. Reorder figures are suggestions; "
        "a manager must approve purchases.</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Dashboard page
# --------------------------------------------------------------------------
def render_dashboard():
    st.title("📊 Inventory dashboard")
    df = inv.enrich(st.session_state.inventory)
    df["stock_value"] = df["stock_on_hand"] * df["unit_price_inr"]
    m = st.columns(4)
    m[0].metric("SKUs", len(df))
    m[1].metric("Stock value", f"₹{df['stock_value'].sum()/1e5:,.1f} L")
    m[2].metric("Out of stock", int((df["status"] == "OUT OF STOCK").sum()))
    m[3].metric("Low stock", int((df["status"] == "LOW").sum()))

    st.markdown("#### Stock status by category")
    pivot = df.pivot_table(index="category", columns="status", values="sku", aggfunc="count", fill_value=0)
    order = [c for c in ["OUT OF STOCK", "LOW", "OK", "OVERSTOCK"] if c in pivot.columns]
    palette = {"OUT OF STOCK": "#d93025", "LOW": "#f29900", "OK": "#1e8e3e", "OVERSTOCK": "#8ab4f8"}
    st.bar_chart(pivot[order], horizontal=True, height=320, color=[palette[c] for c in order])

    st.markdown("#### All products")
    cats = ["All"] + sorted(df["category"].unique())
    f1, f2 = st.columns([1, 1])
    cat = f1.selectbox("Category", cats)
    status = f2.multiselect("Status", ["OUT OF STOCK", "LOW", "OK", "OVERSTOCK"])
    view = df if cat == "All" else df[df["category"] == cat]
    if status:
        view = view[view["status"].isin(status)]
    show = view[["sku", "product", "category", "unit_price_inr", "stock_on_hand", "reorder_level",
                 "units_sold_last_30d", "days_of_cover", "lead_time_days", "suggested_order_qty", "status"]].copy()
    show["days_of_cover"] = show["days_of_cover"].replace(float("inf"), None)

    def color(s):
        return ["background-color:#fde2e1" if v == "OUT OF STOCK" else
                "background-color:#fff1d6" if v == "LOW" else
                "background-color:#e7f0ff" if v == "OVERSTOCK" else "" for v in s]

    st.dataframe(show.style.apply(color, subset=["status"]), width="stretch", hide_index=True,
                 column_config={"sku": "SKU", "product": "Product", "category": "Category",
                                "unit_price_inr": st.column_config.NumberColumn("Price (₹)", format="%d"),
                                "stock_on_hand": "In stock", "reorder_level": "Reorder level",
                                "units_sold_last_30d": "Sold (30d)", "lead_time_days": "Lead time (d)",
                                "suggested_order_qty": "Suggested order", "status": "Status",
                                "days_of_cover": st.column_config.NumberColumn("Days of cover", format="%.1f")})
    buf = io.StringIO()
    show.to_csv(buf, index=False)
    st.download_button("Download this view (CSV)", buf.getvalue(), "inventory_view.csv", "text/csv")
    st.caption("LOW = stock ≤ reorder level, or days of cover < supplier lead time. "
               "OVERSTOCK = more than 90 days of cover. Suggested qty = daily sales × (lead time + 30 days) "
               "+ 7-day safety stock − stock on hand.")


# --------------------------------------------------------------------------
# Chat page
# --------------------------------------------------------------------------
SUGGESTIONS = [
    "Which items are running low?",
    "Do we have the iPhone 15 in stock?",
    "What should I reorder this week?",
    "Give me an inventory summary",
]


def answer(user_text: str) -> tuple[str, list[str], str]:
    """Returns (reply, tool_calls, mode)."""
    bot = get_bot()
    if bot and not st.session_state.ai_error:
        try:
            reply, tools = bot.ask(user_text)
            return reply, tools, "ai"
        except Exception as e:  # noqa: BLE001  API down / quota / garbage output
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                reason = "Gemini free-tier quota reached"
            elif "API key" in msg or "PERMISSION_DENIED" in msg or "401" in msg or "403" in msg:
                reason = "Gemini API key rejected"
            else:
                reason = "Gemini API unavailable"
            st.session_state.ai_error = f"{reason}. Switched to basic mode. Details: {msg[:200]}"
            reply, tools = engine.basic_reply(user_text)
            return (f"_⚠️ {reason}, so this answer comes from basic mode (no AI)._\n\n" + reply), tools, "basic"
    reply, tools = engine.basic_reply(user_text)
    return reply, tools, "basic"


def render_chat():
    st.title("💬 StockSense")
    st.caption("AI stock-query assistant for store staff · ask about availability, low stock and reorders")

    if not st.session_state.messages:
        with st.chat_message("assistant", avatar="📦"):
            st.markdown("Hi! I'm **StockSense**, an AI assistant for this store's inventory. "
                        "I can check stock for any product, flag items running low, and suggest "
                        "reorder quantities based on recent sales. What do you need?")
        cols = st.columns(len(SUGGESTIONS))
        for c, s in zip(cols, SUGGESTIONS):
            if c.button(s, width="stretch"):
                st.session_state.pending = s
                st.rerun()

    for m in st.session_state.messages:
        with st.chat_message(m["role"], avatar="📦" if m["role"] == "assistant" else "🧑‍💼"):
            st.markdown(m["content"])
            if m.get("tools"):
                with st.expander("🔍 Data looked up", expanded=False):
                    for t in m["tools"]:
                        st.code(t, language=None)
                    st.caption("Mode: " + ("Gemini AI" if m.get("mode") == "ai" else "basic (no AI)"))

    typed = st.chat_input("Ask about stock, e.g. 'How many Samsung TVs do we have?'")
    user_text = typed or st.session_state.pending
    st.session_state.pending = None
    if not user_text:
        return

    # Double-submit guard: ignore an identical message sent within 3 seconds.
    last_text, last_time = st.session_state.last_submit
    if user_text == last_text and time.time() - last_time < 3:
        return
    st.session_state.last_submit = (user_text, time.time())

    n_user = sum(1 for m in st.session_state.messages if m["role"] == "user")
    if n_user >= MAX_MESSAGES_PER_SESSION:
        st.warning(f"Session limit of {MAX_MESSAGES_PER_SESSION} questions reached. "
                   "Use 'Clear conversation' in the sidebar to start again.")
        return

    clean, err = engine.validate_user_input(user_text)
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(user_text if clean is None else clean)
    if err:
        with st.chat_message("assistant", avatar="📦"):
            st.markdown(err)
        return

    st.session_state.messages.append({"role": "user", "content": clean})
    with st.chat_message("assistant", avatar="📦"):
        with st.spinner("Checking inventory..."):
            reply, tools, mode = answer(clean)
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply, "tools": tools, "mode": mode})
    st.rerun()


if page.startswith("📊"):
    render_dashboard()
else:
    render_chat()
