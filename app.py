import os
import json
from typing import Dict, Any, List, Tuple

import streamlit as st
import pandas as pd
import altair as alt

# Optional: real AI insights
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# ----------------------------
# Helpers
# ----------------------------

DEFAULT_NEEDS = [
    {"Category": "Rent / mortgage", "Monthly $": 550.0},
    {"Category": "Utilities", "Monthly $": 180.0},
    {"Category": "Groceries", "Monthly $": 400.0},
    {"Category": "Transportation", "Monthly $": 220.0},
    {"Category": "Insurance", "Monthly $": 160.0},
    {"Category": "Debt payments", "Monthly $": 250.0},
]

DEFAULT_WANTS = [
    {"Category": "Dining out", "Monthly $": 180.0},
    {"Category": "Subscriptions", "Monthly $": 45.0},
    {"Category": "Entertainment", "Monthly $": 60.0},
    {"Category": "Other wants", "Monthly $": 80.0},
]


def init_state():
    if "needs_df" not in st.session_state:
        st.session_state["needs_df"] = pd.DataFrame(DEFAULT_NEEDS)
    if "wants_df" not in st.session_state:
        st.session_state["wants_df"] = pd.DataFrame(DEFAULT_WANTS)
    if "ai_time_drain_bullets" not in st.session_state:
        st.session_state["ai_time_drain_bullets"] = []


def clean_budget_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Category" not in df.columns:
        df["Category"] = ""
    if "Monthly $" not in df.columns:
        df["Monthly $"] = 0.0

    df["Category"] = df["Category"].astype(str).str.strip()
    df["Monthly $"] = pd.to_numeric(df["Monthly $"], errors="coerce").fillna(0.0)
    df["Monthly $"] = df["Monthly $"].clip(lower=0.0)

    # Remove blank categories
    df = df[df["Category"] != ""].reset_index(drop=True)
    return df


def effective_hourly_rate(pay_type: str, hourly_rate: float, annual_salary: float, hours_per_week: float) -> float:
    if pay_type == "Hourly":
        return max(float(hourly_rate), 0.01)
    annual_hours = max(float(hours_per_week), 0.01) * 52.0
    return max(float(annual_salary) / annual_hours, 0.01)


def monthly_income(pay_type: str, hourly_rate: float, annual_salary: float, hours_per_week: float) -> float:
    if pay_type == "Hourly":
        weekly_income = float(hourly_rate) * float(hours_per_week)
        return weekly_income * 52.0 / 12.0
    return float(annual_salary) / 12.0


def monthly_work_hours(hours_per_week: float) -> float:
    return float(hours_per_week) * 52.0 / 12.0


def hours_equivalent(amount: float, hourly: float) -> float:
    return float(amount) / max(float(hourly), 0.01)


def sums(inc: float, needs_total: float, wants_total: float) -> Dict[str, float]:
    return {
        "income": inc,
        "needs": needs_total,
        "wants": wants_total,
        "leftover": inc - needs_total - wants_total,
    }


def build_breakdown_df(needs_df: pd.DataFrame, wants_df: pd.DataFrame, hourly: float, income_m: float) -> pd.DataFrame:
    n = needs_df.copy()
    n["Group"] = "Needs"
    w = wants_df.copy()
    w["Group"] = "Wants"
    df = pd.concat([n, w], ignore_index=True)

    df["Hours of work"] = df["Monthly $"].apply(lambda x: hours_equivalent(x, hourly))
    df["Share of income"] = df["Monthly $"].apply(lambda x: (x / income_m) if income_m > 0 else 0.0)
    df = df[["Group", "Category", "Monthly $", "Hours of work", "Share of income"]]
    return df


def top_time_drains(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    out = df.sort_values("Hours of work", ascending=False).head(n).copy()
    out["Monthly $"] = out["Monthly $"].round(0)
    out["Hours of work"] = out["Hours of work"].round(1)
    return out[["Group", "Category", "Monthly $", "Hours of work"]]


def goal_metrics(goal_cost: float, hourly: float, leftover: float) -> Dict[str, float]:
    goal_cost = max(float(goal_cost), 0.0)
    goal_hours = hours_equivalent(goal_cost, hourly)

    if leftover > 0:
        months_to_goal = goal_cost / leftover
        weeks_to_goal = months_to_goal * (52.0 / 12.0)
    else:
        months_to_goal = float("inf")
        weeks_to_goal = float("inf")

    return {"goal_hours": goal_hours, "months_to_goal": months_to_goal, "weeks_to_goal": weeks_to_goal}


# ----------------------------
# Scenarios (deterministic)
# ----------------------------

def propose_scenarios(needs_df: pd.DataFrame, wants_df: pd.DataFrame, leftover: float) -> List[Dict[str, Any]]:
    # Find largest want category
    top_want_row = None
    if len(wants_df) > 0:
        top_want_row = wants_df.sort_values("Monthly $", ascending=False).head(1).to_dict("records")[0]

    scenarios: List[Dict[str, Any]] = []

    if leftover < 0:
        scenarios.append(
            {
                "title": "Stabilize: cut total wants by 15%",
                "type": "cut_total_wants_pct",
                "value": 0.15,
                "why": "Spending exceeds income; stabilize by reducing discretionary spending first.",
            }
        )

    if top_want_row and float(top_want_row["Monthly $"]) > 0:
        scenarios.append(
            {
                "title": f"Reduce {top_want_row['Category']} by 25%",
                "type": "cut_one_want_pct",
                "category": str(top_want_row["Category"]),
                "value": 0.25,
                "why": "Targets the largest discretionary line item for the biggest time savings.",
            }
        )

    scenarios.append(
        {
            "title": "Add 2 work hours per week",
            "type": "add_hours_per_week",
            "value": 2.0,
            "why": "Increases monthly capacity without changing expenses.",
        }
    )

    # Try a low-friction trim if there is something that looks like subscriptions
    sub = wants_df[wants_df["Category"].str.lower().str.contains("sub", na=False)]
    if len(sub) > 0 and float(sub["Monthly $"].sum()) >= 15:
        scenarios.append(
            {
                "title": "Trim subscriptions by $15 per month",
                "type": "delta_wants_total",
                "value": -15.0,
                "why": "Low friction reduction that improves cash flow immediately.",
            }
        )
    else:
        scenarios.append(
            {
                "title": "Reduce wants by $25 per month",
                "type": "delta_wants_total",
                "value": -25.0,
                "why": "Small recurring reduction that adds up without touching necessities.",
            }
        )

    return scenarios[:3]


def apply_scenario(
    pay_type: str,
    hourly_rate: float,
    annual_salary: float,
    hours_per_week: float,
    needs_df: pd.DataFrame,
    wants_df: pd.DataFrame,
    scenario: Dict[str, Any],
) -> Tuple[float, float, float, pd.DataFrame, pd.DataFrame]:
    # Returns updated (hourly_rate, annual_salary, hours_per_week, needs_df, wants_df)
    needs_df2 = needs_df.copy()
    wants_df2 = wants_df.copy()

    t = scenario.get("type")
    val = float(scenario.get("value", 0.0))

    if t == "add_hours_per_week":
        hours_per_week = max(0.0, float(hours_per_week) + val)

    if t == "cut_total_wants_pct":
        pct = max(0.0, min(val, 1.0))
        wants_df2["Monthly $"] = wants_df2["Monthly $"] * (1 - pct)

    if t == "cut_one_want_pct":
        pct = max(0.0, min(val, 1.0))
        cat = str(scenario.get("category", "")).strip()
        mask = wants_df2["Category"].astype(str).str.strip() == cat
        wants_df2.loc[mask, "Monthly $"] = wants_df2.loc[mask, "Monthly $"] * (1 - pct)

    if t == "delta_wants_total":
        delta = val
        # Apply delta to largest want category to keep the change interpretable
        if len(wants_df2) > 0:
            idx = wants_df2["Monthly $"].idxmax()
            wants_df2.loc[idx, "Monthly $"] = max(0.0, float(wants_df2.loc[idx, "Monthly $"]) + delta)

    wants_df2 = clean_budget_df(wants_df2)
    needs_df2 = clean_budget_df(needs_df2)
    return hourly_rate, annual_salary, hours_per_week, needs_df2, wants_df2


# ----------------------------
# AI insights (real + fallback)
# ----------------------------

def ai_insights(
    model: str,
    snapshot: Dict[str, float],
    top3: List[Dict[str, Any]],
    scenarios: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    hr = snapshot["effective_hourly_rate"]
    fallback_lines = [
        f"Effective hourly rate: ${hr:,.2f}. Estimated monthly income: ${snapshot['income']:,.0f}.",
        f"Needs: ${snapshot['needs']:,.0f} ({snapshot['needs_pct']:.0f}%). Wants: ${snapshot['wants']:,.0f} ({snapshot['wants_pct']:.0f}%).",
        f"Leftover: ${snapshot['leftover']:,.0f}. Wants represent {snapshot['wants_hours']:.1f} work-hours per month.",
        "Top time drains: " + ", ".join([f"{x['Category']} ({x['Hours of work']:.1f}h)" for x in top3]),
    ]
    fallback_text = "\n".join(fallback_lines)

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or OpenAI is None:
        st.session_state["ai_time_drain_bullets"] = []
        return fallback_text, scenarios

    try:
        client = OpenAI()
        prompt = {
            "role": "user",
            "content": (
                "You are a concise, analytical personal finance analyst.\n"
                "Return JSON only.\n\n"
                "Write:\n"
                "1) summary: 4 to 6 sentences focusing on opportunity cost in time terms.\n"
                "2) top_time_drains: 3 bullets (short phrases) explaining why the top categories matter.\n"
                "3) ranked_scenarios: rank provided scenarios best to worst with one-sentence rationale each.\n\n"
                f"Monthly snapshot: {json.dumps(snapshot, indent=2)}\n"
                f"Top categories by hours: {json.dumps(top3, indent=2)}\n"
                f"Scenarios: {json.dumps([{'title': s['title'], 'why': s['why'], 'type': s.get('type'), 'value': s.get('value'), 'category': s.get('category')} for s in scenarios], indent=2)}\n"
            ),
        }

        resp = client.responses.create(model=model, input=[prompt])
        text = (getattr(resp, "output_text", "") or "").strip()
        data = json.loads(text)

        summary = str(data.get("summary", "")).strip() or fallback_text
        bullets = data.get("top_time_drains", [])
        if isinstance(bullets, list):
            st.session_state["ai_time_drain_bullets"] = [str(b).strip() for b in bullets if str(b).strip()]
        else:
            st.session_state["ai_time_drain_bullets"] = []

        ranked = data.get("ranked_scenarios", [])
        rationale_map = {}
        if isinstance(ranked, list):
            for r in ranked:
                if isinstance(r, dict) and "title" in r:
                    rationale_map[str(r["title"]).strip()] = str(r.get("rationale", "")).strip()

        out = []
        for s in scenarios:
            s2 = dict(s)
            if s2["title"] in rationale_map and rationale_map[s2["title"]]:
                s2["ai_rationale"] = rationale_map[s2["title"]]
            out.append(s2)

        return summary, out

    except Exception:
        st.session_state["ai_time_drain_bullets"] = []
        return fallback_text, scenarios


# ----------------------------
# App
# ----------------------------

st.set_page_config(page_title="Opportunity Cost Visualizer", layout="wide")
init_state()

st.title("Opportunity Cost Visualizer")
st.caption("Visualize income as time, then use AI-generated hypotheticals to reduce high-cost spending patterns.")

tab_setup, tab_dash, tab_insights, tab_scenarios = st.tabs(["Setup", "Dashboard", "Insights", "Scenarios"])

with tab_setup:
    st.subheader("1) Pay and time inputs")

    c1, c2, c3 = st.columns([1.1, 1.2, 1.2], gap="large")
    with c1:
        pay_type = st.radio("Pay type", ["Hourly", "Salary"], horizontal=True)
    with c2:
        if pay_type == "Hourly":
            hourly_rate = st.number_input("Hourly rate ($)", min_value=0.0, value=22.0, step=0.5)
            annual_salary = 0.0
        else:
            annual_salary = st.number_input("Annual salary ($)", min_value=0.0, value=76000.0, step=500.0)
            hourly_rate = 0.0
    with c3:
        hours_per_week = st.number_input("Actual hours worked per week", min_value=0.0, value=44.0, step=1.0)

    st.divider()
    st.subheader("2) Customize your monthly Needs and Wants")
    st.caption("Add rows, rename categories, or adjust amounts. Blank categories are ignored.")

    ncol, wcol = st.columns(2, gap="large")

    with ncol:
        st.markdown("### Needs (editable)")
        needs_edit = st.data_editor(
            st.session_state["needs_df"],
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Category": st.column_config.TextColumn(required=True),
                "Monthly $": st.column_config.NumberColumn(min_value=0.0, step=5.0, format="%.2f"),
            },
        )
        st.session_state["needs_df"] = clean_budget_df(needs_edit)

    with wcol:
        st.markdown("### Wants (editable)")
        wants_edit = st.data_editor(
            st.session_state["wants_df"],
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Category": st.column_config.TextColumn(required=True),
                "Monthly $": st.column_config.NumberColumn(min_value=0.0, step=5.0, format="%.2f"),
            },
        )
        st.session_state["wants_df"] = clean_budget_df(wants_edit)

    st.divider()
    st.subheader("3) Goal mode")
    g1, g2 = st.columns([1.6, 1.0], gap="large")
    with g1:
        goal_name = st.text_input("Goal name", value="New laptop")
    with g2:
        goal_cost = st.number_input("Goal cost ($)", min_value=0.0, value=900.0, step=25.0)

    st.divider()
    st.subheader("AI settings")
    model = st.text_input("OpenAI model", value="gpt-5.2")
    st.caption("Keep your API key outside the code. Use export OPENAI_API_KEY=... in the terminal.")

# Use the latest values from state
needs_df = st.session_state["needs_df"]
wants_df = st.session_state["wants_df"]

# Compute baseline
income_m = monthly_income(pay_type, hourly_rate, annual_salary, hours_per_week)
hr = effective_hourly_rate(pay_type, hourly_rate, annual_salary, hours_per_week)
work_hours = monthly_work_hours(hours_per_week)

needs_total = float(needs_df["Monthly $"].sum()) if len(needs_df) > 0 else 0.0
wants_total = float(wants_df["Monthly $"].sum()) if len(wants_df) > 0 else 0.0
base = sums(income_m, needs_total, wants_total)

breakdown = build_breakdown_df(needs_df, wants_df, hr, income_m)
top3_df = top_time_drains(breakdown, n=3)
top3_records = top3_df.to_dict(orient="records")

snapshot = {
    "income": base["income"],
    "needs": base["needs"],
    "wants": base["wants"],
    "leftover": base["leftover"],
    "effective_hourly_rate": hr,
    "needs_pct": (base["needs"] / base["income"] * 100) if base["income"] > 0 else 0.0,
    "wants_pct": (base["wants"] / base["income"] * 100) if base["income"] > 0 else 0.0,
    "wants_hours": hours_equivalent(base["wants"], hr),
}

scenarios = propose_scenarios(needs_df, wants_df, base["leftover"])
analysis_text, scenarios_out = ai_insights(model=model, snapshot=snapshot, top3=top3_records, scenarios=scenarios)
ai_bullets = st.session_state.get("ai_time_drain_bullets", [])

with tab_dash:
    left, mid, right = st.columns([1.2, 1.6, 1.2], gap="large")

    with left:
        st.subheader("Key metrics")
        st.metric("Monthly income (estimated)", f"${base['income']:,.0f}")
        st.metric("Effective hourly rate", f"${hr:,.2f}")
        st.metric("Monthly work hours", f"{work_hours:,.1f} hrs")
        st.metric("Leftover after needs + wants", f"${base['leftover']:,.0f}")

        st.subheader("Goal mode")
        gm = goal_metrics(goal_cost, hr, base["leftover"])
        st.write(f"Goal: **{goal_name}** (${goal_cost:,.0f})")
        st.write(f"Time cost: **{gm['goal_hours']:.1f} hours** of work")
        if gm["weeks_to_goal"] != float("inf"):
            st.write(f"Estimated time to goal (using leftover): **{gm['weeks_to_goal']:.1f} weeks**")
        else:
            st.warning("Leftover is not positive, so the goal cannot be funded from current leftover.")

    with mid:
        st.subheader("Time budget breakdown")
        chart_df = pd.DataFrame(
            [
                {"Bucket": "Needs", "Hours": float(hours_equivalent(base["needs"], hr))},
                {"Bucket": "Wants", "Hours": float(hours_equivalent(base["wants"], hr))},
                {"Bucket": "Unallocated", "Hours": float(max(0.0, work_hours - hours_equivalent(base["needs"] + base["wants"], hr)))},
            ]
        )

        chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("Hours:Q", title="Hours of work per month"),
                y=alt.Y("Bucket:N", title=""),
                tooltip=["Bucket", alt.Tooltip("Hours:Q", format=",.1f")],
            )
            .properties(height=180)
        )
        st.altair_chart(chart, use_container_width=True)

        st.subheader("Top time drains")
        st.dataframe(top3_df, use_container_width=True, hide_index=True)

    with right:
        st.subheader("AI insights (summary)")
        st.text_area("Summary", analysis_text, height=220)

with tab_insights:
    col1, col2 = st.columns([1.3, 1.7], gap="large")

    with col1:
        st.subheader("AI: why the top categories matter")
        if ai_bullets:
            for b in ai_bullets[:3]:
                st.write(f"- {b}")
        else:
            st.write("- High-hour categories are the main opportunity cost drivers.")
            st.write("- Small changes here typically produce the largest time savings.")
            st.write("- Optimize these before focusing on small items.")

        st.subheader("Opportunity cost view")
        needs_hours = hours_equivalent(base["needs"], hr)
        wants_hours = hours_equivalent(base["wants"], hr)
        st.write(f"Needs consume about **{needs_hours:,.1f}** work-hours per month.")
        st.write(f"Wants consume about **{wants_hours:,.1f}** work-hours per month.")

    with col2:
        st.subheader("Detailed categories")
        df_show = breakdown.copy()
        df_show["Share of income"] = (df_show["Share of income"] * 100).round(1).astype(str) + "%"
        st.dataframe(df_show, use_container_width=True, hide_index=True)

with tab_scenarios:
    st.subheader("Suggested hypotheticals")
    st.caption("Select a scenario to compare it against your baseline.")

    titles = [s["title"] for s in scenarios_out]
    if not titles:
        st.info("Add at least one Want category to generate scenarios.")
    else:
        choice = st.radio("Scenarios", titles, index=0, horizontal=False)
        picked = next(s for s in scenarios_out if s["title"] == choice)

        hr2, sal2, hpw2, needs2, wants2 = apply_scenario(
            pay_type, hourly_rate, annual_salary, hours_per_week, needs_df, wants_df, picked
        )

        income2 = monthly_income(pay_type, hr2, sal2, hpw2)
        eff_hr2 = effective_hourly_rate(pay_type, hr2, sal2, hpw2)
        needs2_total = float(needs2["Monthly $"].sum()) if len(needs2) > 0 else 0.0
        wants2_total = float(wants2["Monthly $"].sum()) if len(wants2) > 0 else 0.0
        updated = sums(income2, needs2_total, wants2_total)

        delta_leftover = updated["leftover"] - base["leftover"]
        delta_wants_hours = hours_equivalent(updated["wants"], eff_hr2) - hours_equivalent(base["wants"], hr)

        c1, c2, c3 = st.columns(3)
        c1.metric("Leftover change", f"${delta_leftover:,.0f}")
        c2.metric("Wants time change", f"{delta_wants_hours:,.1f} hrs")
        c3.metric("Adjusted monthly income", f"${updated['income']:,.0f}")

        st.subheader("Rationale")
        st.write(picked.get("ai_rationale") or picked.get("why") or "Scenario rationale unavailable.")

        comp_df = pd.DataFrame(
            [
                {"Scenario": "Base", "Needs (hrs)": hours_equivalent(base["needs"], hr), "Wants (hrs)": hours_equivalent(base["wants"], hr)},
                {"Scenario": "Adjusted", "Needs (hrs)": hours_equivalent(updated["needs"], eff_hr2), "Wants (hrs)": hours_equivalent(updated["wants"], eff_hr2)},
            ]
        ).melt(id_vars=["Scenario"], var_name="Bucket", value_name="Hours")

        comp_chart = (
            alt.Chart(comp_df)
            .mark_bar()
            .encode(
                x=alt.X("Hours:Q", title="Hours"),
                y=alt.Y("Scenario:N", title=""),
                color=alt.Color("Bucket:N", legend=alt.Legend(title="")),
                tooltip=["Scenario", "Bucket", alt.Tooltip("Hours:Q", format=",.1f")],
            )
            .properties(height=220)
        )
        st.altair_chart(comp_chart, use_container_width=True)
