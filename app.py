import os
import json
from dataclasses import dataclass, asdict
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
# Core math
# ----------------------------

@dataclass
class Inputs:
    pay_type: str  # "Hourly" or "Salary"
    hourly_rate: float
    annual_salary: float
    hours_per_week: float
    pay_periods_per_year: int  # 12 monthly, 26 biweekly, 52 weekly, 24 semimonthly

    # Expenses (monthly)
    rent: float
    utilities: float
    groceries: float
    transport: float
    insurance: float
    debt: float

    # Wants (monthly)
    dining: float
    subscriptions: float
    entertainment: float
    other_wants: float


def effective_hourly_rate(i: Inputs) -> float:
    if i.pay_type == "Hourly":
        return max(i.hourly_rate, 0.01)
    # Convert salary to hourly using user-entered hours/week
    annual_hours = max(i.hours_per_week, 0.01) * 52.0
    return max(i.annual_salary / annual_hours, 0.01)


def monthly_income(i: Inputs) -> float:
    if i.pay_type == "Hourly":
        weekly_income = i.hourly_rate * i.hours_per_week
        return weekly_income * 52.0 / 12.0
    return i.annual_salary / 12.0


def monthly_work_hours(i: Inputs) -> float:
    # Use actual hours/week as the base "time budget" for income
    return i.hours_per_week * 52.0 / 12.0


def sums(i: Inputs) -> Dict[str, float]:
    needs = i.rent + i.utilities + i.groceries + i.transport + i.insurance + i.debt
    wants = i.dining + i.subscriptions + i.entertainment + i.other_wants
    inc = monthly_income(i)
    return {"income": inc, "needs": needs, "wants": wants, "leftover": inc - needs - wants}


def hours_equivalent(amount: float, hourly: float) -> float:
    return amount / max(hourly, 0.01)


def build_breakdown_df(i: Inputs) -> pd.DataFrame:
    hr = effective_hourly_rate(i)
    inc = monthly_income(i)
    items = [
        ("Needs", "Rent", i.rent),
        ("Needs", "Utilities", i.utilities),
        ("Needs", "Groceries", i.groceries),
        ("Needs", "Transport", i.transport),
        ("Needs", "Insurance", i.insurance),
        ("Needs", "Debt", i.debt),
        ("Wants", "Dining", i.dining),
        ("Wants", "Subscriptions", i.subscriptions),
        ("Wants", "Entertainment", i.entertainment),
        ("Wants", "Other wants", i.other_wants),
    ]
    rows = []
    for group, name, amt in items:
        rows.append(
            {
                "Group": group,
                "Category": name,
                "Monthly $": float(amt),
                "Hours of work": float(hours_equivalent(amt, hr)),
                "Share of income": float(amt / inc) if inc > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


# ----------------------------
# Scenario generation (deterministic)
# ----------------------------

def propose_scenarios(i: Inputs) -> List[Dict[str, Any]]:
    s = sums(i)
    inc = s["income"]

    # Identify top wants line item
    wants_map = {
        "Dining": i.dining,
        "Subscriptions": i.subscriptions,
        "Entertainment": i.entertainment,
        "Other wants": i.other_wants,
    }
    top_want = max(wants_map.items(), key=lambda x: x[1])[0]
    top_want_amt = wants_map[top_want]

    scenarios = []

    # 1) Cut top want by 25%
    if top_want_amt > 0:
        scenarios.append(
            {
                "title": f"Reduce {top_want} by 25%",
                "changes": {"cut_wants": {top_want: 0.25}},
                "why": "Targets the highest discretionary line item for maximum impact with minimal complexity.",
            }
        )

    # 2) Add 2 hours/week work
    scenarios.append(
        {
            "title": "Add 2 work hours per week",
            "changes": {"add_hours_per_week": 2.0},
            "why": "Increases capacity without changing expenses, useful when essentials consume most income.",
        }
    )

    # 3) Cancel or reduce subscriptions by $15 if meaningful
    if i.subscriptions >= 15:
        scenarios.append(
            {
                "title": "Trim subscriptions by $15 per month",
                "changes": {"set_subscriptions_delta": -15.0},
                "why": "Low friction change that improves cash flow immediately.",
            }
        )
    else:
        scenarios.append(
            {
                "title": "Reduce dining by $25 per month",
                "changes": {"set_dining_delta": -25.0},
                "why": "Small recurring reduction that adds up without changing necessities.",
            }
        )

    # If income is negative leftover, add a stabilizing scenario
    if s["leftover"] < 0:
        scenarios.insert(
            0,
            {
                "title": "Stabilize: cut total wants by 15%",
                "changes": {"cut_total_wants_pct": 0.15},
                "why": "Brings spending back within income when the current month runs a deficit.",
            },
        )

    # Keep top 3
    return scenarios[:3]


def apply_scenario(i: Inputs, scenario: Dict[str, Any]) -> Inputs:
    j = Inputs(**asdict(i))
    ch = scenario.get("changes", {})

    # Add work hours
    if "add_hours_per_week" in ch:
        j.hours_per_week = max(0.0, j.hours_per_week + float(ch["add_hours_per_week"]))

    # Cut total wants by percent
    if "cut_total_wants_pct" in ch:
        pct = float(ch["cut_total_wants_pct"])
        j.dining *= (1 - pct)
        j.subscriptions *= (1 - pct)
        j.entertainment *= (1 - pct)
        j.other_wants *= (1 - pct)

    # Cut specific want by percent
    if "cut_wants" in ch:
        for k, pct in ch["cut_wants"].items():
            pct = float(pct)
            if k == "Dining":
                j.dining *= (1 - pct)
            elif k == "Subscriptions":
                j.subscriptions *= (1 - pct)
            elif k == "Entertainment":
                j.entertainment *= (1 - pct)
            elif k == "Other wants":
                j.other_wants *= (1 - pct)

    # Direct deltas
    if "set_subscriptions_delta" in ch:
        j.subscriptions = max(0.0, j.subscriptions + float(ch["set_subscriptions_delta"]))
    if "set_dining_delta" in ch:
        j.dining = max(0.0, j.dining + float(ch["set_dining_delta"]))

    return j


# ----------------------------
# AI insights (real + fallback)
# ----------------------------

def ai_insights(i: Inputs, scenarios: List[Dict[str, Any]], model: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Returns (analysis_text, scenarios_out)
    scenarios_out optionally enhanced by the model, but always usable.
    """
    # Fallback text
    base = sums(i)
    hr = effective_hourly_rate(i)

    fallback_lines = [
        f"Effective hourly rate: ${hr:,.2f}. Estimated monthly income: ${base['income']:,.0f}.",
        f"Needs: ${base['needs']:,.0f} ({(base['needs']/base['income']*100) if base['income']>0 else 0:,.0f}%). "
        f"Wants: ${base['wants']:,.0f} ({(base['wants']/base['income']*100) if base['income']>0 else 0:,.0f}%).",
        f"Leftover: ${base['leftover']:,.0f}. Opportunity cost framing: ${base['wants']:,.0f} of wants equals {hours_equivalent(base['wants'], hr):,.1f} work-hours per month.",
    ]
    fallback_text = "\n".join(fallback_lines)

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or OpenAI is None:
        return fallback_text, scenarios

    try:
        client = OpenAI()

        # Ask for short, analytical output, and a JSON scenario ranking
        prompt = {
            "role": "user",
            "content": (
                "You are a concise, analytical personal finance analyst. "
                "Given the user's monthly income and expense breakdown, write:\n"
                "1) A 4 to 6 sentence analytical summary focusing on opportunity cost in time terms.\n"
                "2) Rank the provided scenarios from best to worst and add a one-sentence rationale for each.\n"
                "Return JSON with keys: summary (string), ranked_scenarios (array of objects with title and rationale).\n\n"
                f"Inputs (monthly): {json.dumps({'income': base['income'], 'needs': base['needs'], 'wants': base['wants'], 'leftover': base['leftover'], 'effective_hourly_rate': hr}, indent=2)}\n"
                f"Scenarios: {json.dumps([{'title': s['title'], 'why': s['why'], 'changes': s['changes']} for s in scenarios], indent=2)}\n"
            ),
        }

        # Use Responses API (recommended in OpenAI docs) and read output_text
        resp = client.responses.create(
            model=model,
            input=[prompt],
        )
        text = (getattr(resp, "output_text", "") or "").strip()
        data = json.loads(text)

        summary = str(data.get("summary", "")).strip() or fallback_text
        ranked = data.get("ranked_scenarios", [])

        # Merge rationales back onto our scenarios by title
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
        return fallback_text, scenarios


# ----------------------------
# UI
# ----------------------------

st.set_page_config(page_title="Opportunity Cost Visualizer", layout="wide")

st.title("Opportunity Cost Visualizer")
st.caption("Visualize income as time, then use AI-generated hypotheticals to reduce high-cost spending patterns.")

with st.sidebar:
    st.header("Inputs")

    pay_type = st.radio("Pay type", ["Hourly", "Salary"], horizontal=True)

    if pay_type == "Hourly":
        hourly_rate = st.number_input("Hourly rate ($)", min_value=0.0, value=22.0, step=0.5)
        annual_salary = 0.0
    else:
        annual_salary = st.number_input("Annual salary ($)", min_value=0.0, value=55000.0, step=500.0)
        hourly_rate = 0.0

    hours_per_week = st.number_input("Actual hours worked per week", min_value=0.0, value=40.0, step=1.0)

    st.divider()
    st.subheader("Monthly necessities")
    rent = st.number_input("Rent / mortgage", min_value=0.0, value=1400.0, step=25.0)
    utilities = st.number_input("Utilities", min_value=0.0, value=180.0, step=10.0)
    groceries = st.number_input("Groceries", min_value=0.0, value=400.0, step=10.0)
    transport = st.number_input("Transportation", min_value=0.0, value=220.0, step=10.0)
    insurance = st.number_input("Insurance", min_value=0.0, value=160.0, step=10.0)
    debt = st.number_input("Debt payments", min_value=0.0, value=250.0, step=10.0)

    st.divider()
    st.subheader("Monthly wants")
    dining = st.number_input("Dining out", min_value=0.0, value=180.0, step=10.0)
    subscriptions = st.number_input("Subscriptions", min_value=0.0, value=45.0, step=5.0)
    entertainment = st.number_input("Entertainment", min_value=0.0, value=60.0, step=5.0)
    other_wants = st.number_input("Other wants", min_value=0.0, value=80.0, step=5.0)

    st.divider()
    st.subheader("AI settings")
    model = st.text_input("OpenAI model", value="gpt-5.2")
    st.caption("If OPENAI_API_KEY is not set, the app uses a built-in analytical fallback.")


inputs = Inputs(
    pay_type=pay_type,
    hourly_rate=float(hourly_rate),
    annual_salary=float(annual_salary),
    hours_per_week=float(hours_per_week),
    pay_periods_per_year=12,
    rent=float(rent),
    utilities=float(utilities),
    groceries=float(groceries),
    transport=float(transport),
    insurance=float(insurance),
    debt=float(debt),
    dining=float(dining),
    subscriptions=float(subscriptions),
    entertainment=float(entertainment),
    other_wants=float(other_wants),
)

base = sums(inputs)
hr = effective_hourly_rate(inputs)
work_hours = monthly_work_hours(inputs)

scenarios = propose_scenarios(inputs)
analysis_text, scenarios_out = ai_insights(inputs, scenarios, model=model)

# Layout
left, mid, right = st.columns([1.2, 1.6, 1.2], gap="large")

with left:
    st.subheader("Key metrics")
    st.metric("Monthly income (estimated)", f"${base['income']:,.0f}")
    st.metric("Effective hourly rate", f"${hr:,.2f}")
    st.metric("Monthly work hours", f"{work_hours:,.1f} hrs")
    st.metric("Leftover after needs + wants", f"${base['leftover']:,.0f}")

    st.subheader("Opportunity cost view")
    wants_hours = hours_equivalent(base["wants"], hr)
    needs_hours = hours_equivalent(base["needs"], hr)
    st.write(f"Needs consume about **{needs_hours:,.1f}** work-hours per month.")
    st.write(f"Wants consume about **{wants_hours:,.1f}** work-hours per month.")

with mid:
    st.subheader("Time budget breakdown")

    # Build chart data
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

    st.subheader("Detailed categories")
    df = build_breakdown_df(inputs)
    df_show = df[["Group", "Category", "Monthly $", "Hours of work", "Share of income"]].copy()
    df_show["Share of income"] = (df_show["Share of income"] * 100).round(1).astype(str) + "%"
    st.dataframe(df_show, use_container_width=True, hide_index=True)

with right:
    st.subheader("AI insights")
    st.text_area("Summary", analysis_text, height=180)

    st.subheader("Suggested hypotheticals")
    st.caption("Click a scenario to see the time-based opportunity cost change.")

    # Scenario selection
    titles = [s["title"] for s in scenarios_out]
    choice = st.radio("Scenarios", titles, index=0)

    picked = next(s for s in scenarios_out if s["title"] == choice)
    updated_inputs = apply_scenario(inputs, picked)
    updated = sums(updated_inputs)
    updated_hr = effective_hourly_rate(updated_inputs)
    updated_work_hours = monthly_work_hours(updated_inputs)

    # Scenario impact
    delta_leftover = updated["leftover"] - base["leftover"]
    delta_wants_hours = hours_equivalent(updated["wants"], updated_hr) - hours_equivalent(base["wants"], hr)

    st.metric("Leftover change", f"${delta_leftover:,.0f}")
    st.metric("Wants time change", f"{delta_wants_hours:,.1f} hrs")

    if picked.get("ai_rationale"):
        st.write("**Rationale**")
        st.write(picked["ai_rationale"])
    else:
        st.write("**Rationale**")
        st.write(picked.get("why", "Scenario rationale unavailable."))

    # Small comparison bar
    comp_df = pd.DataFrame(
        [
            {"Scenario": "Base", "Needs (hrs)": hours_equivalent(base["needs"], hr), "Wants (hrs)": hours_equivalent(base["wants"], hr)},
            {"Scenario": "Adjusted", "Needs (hrs)": hours_equivalent(updated["needs"], updated_hr), "Wants (hrs)": hours_equivalent(updated["wants"], updated_hr)},
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
        .properties(height=160)
    )
    st.altair_chart(comp_chart, use_container_width=True)
