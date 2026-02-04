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


# ============================================================================
# CONSTANTS & DEFAULTS
# ============================================================================

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


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_state():
    """Initialize all session state variables"""
    if "needs_df" not in st.session_state:
        st.session_state["needs_df"] = pd.DataFrame(DEFAULT_NEEDS)
    if "wants_df" not in st.session_state:
        st.session_state["wants_df"] = pd.DataFrame(DEFAULT_WANTS)
    if "ai_time_drain_bullets" not in st.session_state:
        st.session_state["ai_time_drain_bullets"] = []
    if "pay_type" not in st.session_state:
        st.session_state["pay_type"] = "Hourly"
    if "hourly_rate" not in st.session_state:
        st.session_state["hourly_rate"] = 22.0
    if "annual_salary" not in st.session_state:
        st.session_state["annual_salary"] = 76000.0
    if "hours_per_week" not in st.session_state:
        st.session_state["hours_per_week"] = 44.0
    if "goal_name" not in st.session_state:
        st.session_state["goal_name"] = "New laptop"
    if "goal_cost" not in st.session_state:
        st.session_state["goal_cost"] = 900.0
    if "ai_model" not in st.session_state:
        st.session_state["ai_model"] = "gpt-4o"  # Fixed to valid model
    if "setup_complete" not in st.session_state:
        st.session_state["setup_complete"] = False


# ============================================================================
# DATA VALIDATION & CLEANING
# ============================================================================

def clean_budget_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate budget DataFrame"""
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


def validate_setup() -> Tuple[bool, str]:
    """Validate that minimum setup is complete"""
    if st.session_state["hours_per_week"] <= 0:
        return False, "⚠️ Please enter hours worked per week"
    
    if st.session_state["pay_type"] == "Hourly" and st.session_state["hourly_rate"] <= 0:
        return False, "⚠️ Please enter your hourly rate"
    
    if st.session_state["pay_type"] == "Salary" and st.session_state["annual_salary"] <= 0:
        return False, "⚠️ Please enter your annual salary"
    
    needs_df = st.session_state["needs_df"]
    wants_df = st.session_state["wants_df"]
    
    if len(needs_df) == 0 and len(wants_df) == 0:
        return False, "⚠️ Please add at least one budget category"
    
    return True, "✅ Setup complete!"


# ============================================================================
# CALCULATION FUNCTIONS
# ============================================================================

def effective_hourly_rate(pay_type: str, hourly_rate: float, annual_salary: float, hours_per_week: float) -> float:
    """Calculate effective hourly rate based on pay type"""
    if pay_type == "Hourly":
        return max(float(hourly_rate), 0.01)
    annual_hours = max(float(hours_per_week), 0.01) * 52.0
    return max(float(annual_salary) / annual_hours, 0.01)


def monthly_income(pay_type: str, hourly_rate: float, annual_salary: float, hours_per_week: float) -> float:
    """Calculate monthly income"""
    if pay_type == "Hourly":
        weekly_income = float(hourly_rate) * float(hours_per_week)
        return weekly_income * 52.0 / 12.0
    return float(annual_salary) / 12.0


def monthly_work_hours(hours_per_week: float) -> float:
    """Calculate monthly work hours"""
    return float(hours_per_week) * 52.0 / 12.0


def hours_equivalent(amount: float, hourly: float) -> float:
    """Convert dollar amount to hours of work"""
    return float(amount) / max(float(hourly), 0.01)


def sums(inc: float, needs_total: float, wants_total: float) -> Dict[str, float]:
    """Calculate summary totals"""
    return {
        "income": inc,
        "needs": needs_total,
        "wants": wants_total,
        "leftover": inc - needs_total - wants_total,
    }


def build_breakdown_df(needs_df: pd.DataFrame, wants_df: pd.DataFrame, hourly: float, income_m: float) -> pd.DataFrame:
    """Build detailed breakdown of all categories"""
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
    """Get top N categories by hours of work"""
    out = df.sort_values("Hours of work", ascending=False).head(n).copy()
    out["Monthly $"] = out["Monthly $"].round(0)
    out["Hours of work"] = out["Hours of work"].round(1)
    return out[["Group", "Category", "Monthly $", "Hours of work"]]


def goal_metrics(goal_cost: float, hourly: float, leftover: float) -> Dict[str, float]:
    """Calculate goal achievement metrics"""
    goal_cost = max(float(goal_cost), 0.0)
    goal_hours = hours_equivalent(goal_cost, hourly)

    if leftover > 0:
        months_to_goal = goal_cost / leftover
        weeks_to_goal = months_to_goal * (52.0 / 12.0)
    else:
        months_to_goal = float("inf")
        weeks_to_goal = float("inf")

    return {"goal_hours": goal_hours, "months_to_goal": months_to_goal, "weeks_to_goal": weeks_to_goal}


# ============================================================================
# SCENARIO GENERATION
# ============================================================================

def propose_scenarios(needs_df: pd.DataFrame, wants_df: pd.DataFrame, leftover: float) -> List[Dict[str, Any]]:
    """Generate personalized scenarios based on budget state"""
    top_want_row = None
    if len(wants_df) > 0:
        top_want_row = wants_df.sort_values("Monthly $", ascending=False).head(1).to_dict("records")[0]

    scenarios: List[Dict[str, Any]] = []

    # Scenario 1: If in deficit, suggest stabilization
    if leftover < 0:
        scenarios.append({
            "title": "Stabilize: cut total wants by 15%",
            "type": "cut_total_wants_pct",
            "value": 0.15,
            "why": "Spending exceeds income; stabilize by reducing discretionary spending first.",
        })

    # Scenario 2: Target largest want category
    if top_want_row and float(top_want_row["Monthly $"]) > 0:
        scenarios.append({
            "title": f"Reduce {top_want_row['Category']} by 25%",
            "type": "cut_one_want_pct",
            "category": str(top_want_row["Category"]),
            "value": 0.25,
            "why": "Targets the largest discretionary line item for the biggest time savings.",
        })

    # Scenario 3: Increase work hours
    scenarios.append({
        "title": "Add 2 work hours per week",
        "type": "add_hours_per_week",
        "value": 2.0,
        "why": "Increases monthly capacity without changing expenses.",
    })

    # Scenario 4: Trim subscriptions or small reduction
    sub = wants_df[wants_df["Category"].str.lower().str.contains("sub", na=False)]
    if len(sub) > 0 and float(sub["Monthly $"].sum()) >= 15:
        scenarios.append({
            "title": "Trim subscriptions by $15 per month",
            "type": "delta_wants_total",
            "value": -15.0,
            "why": "Low friction reduction that improves cash flow immediately.",
        })
    else:
        scenarios.append({
            "title": "Reduce wants by $25 per month",
            "type": "delta_wants_total",
            "value": -25.0,
            "why": "Small recurring reduction that adds up without touching necessities.",
        })

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
    """Apply a scenario transformation to the budget"""
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
        if len(wants_df2) > 0:
            idx = wants_df2["Monthly $"].idxmax()
            wants_df2.loc[idx, "Monthly $"] = max(0.0, float(wants_df2.loc[idx, "Monthly $"]) + delta)

    wants_df2 = clean_budget_df(wants_df2)
    needs_df2 = clean_budget_df(needs_df2)
    return hourly_rate, annual_salary, hours_per_week, needs_df2, wants_df2


# ============================================================================
# AI INSIGHTS
# ============================================================================

def ai_insights(
    model: str,
    snapshot: Dict[str, float],
    top3: List[Dict[str, Any]],
    scenarios: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Generate AI-powered insights using OpenAI API"""
    hr = snapshot["effective_hourly_rate"]
    
    # Fallback response if AI is unavailable
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
        client = OpenAI(api_key=api_key)
        
        prompt_content = (
            "You are a concise, analytical personal finance analyst.\n"
            "Return JSON only with this exact structure:\n"
            "{\n"
            '  "summary": "4-6 sentences focusing on opportunity cost in time terms",\n'
            '  "top_time_drains": ["bullet 1", "bullet 2", "bullet 3"],\n'
            '  "ranked_scenarios": [\n'
            '    {"title": "scenario title", "rationale": "one sentence why this is ranked here"}\n'
            "  ]\n"
            "}\n\n"
            f"Monthly snapshot: {json.dumps(snapshot, indent=2)}\n"
            f"Top categories by hours: {json.dumps(top3, indent=2)}\n"
            f"Scenarios: {json.dumps([{'title': s['title'], 'why': s['why'], 'type': s.get('type'), 'value': s.get('value'), 'category': s.get('category')} for s in scenarios], indent=2)}\n"
        )

        # Fixed: Use correct OpenAI API call
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a financial analyst. Return only valid JSON."},
                {"role": "user", "content": prompt_content}
            ],
            temperature=0.7,
        )
        
        text = response.choices[0].message.content.strip()
        
        # Try to extract JSON if wrapped in markdown code blocks
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        
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

    except Exception as e:
        st.warning(f"AI insights unavailable: {str(e)}")
        st.session_state["ai_time_drain_bullets"] = []
        return fallback_text, scenarios


# ============================================================================
# UI RENDERING FUNCTIONS
# ============================================================================

def render_setup_tab():
    """Render the setup/input tab"""
    st.header("💼 Income & Budget Setup")
    st.markdown("---")
    
    # Pay and time inputs
    st.subheader("1️⃣ Income Information")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        pay_type = st.radio("Pay type", ["Hourly", "Salary"], 
                           key="pay_type_input",
                           index=0 if st.session_state["pay_type"] == "Hourly" else 1)
        st.session_state["pay_type"] = pay_type
    
    with col2:
        if pay_type == "Hourly":
            hourly_rate = st.number_input("Hourly rate ($)", 
                                         min_value=0.0, 
                                         value=st.session_state["hourly_rate"], 
                                         step=0.5,
                                         key="hourly_input")
            st.session_state["hourly_rate"] = hourly_rate
            st.session_state["annual_salary"] = 0.0
        else:
            annual_salary = st.number_input("Annual salary ($)", 
                                           min_value=0.0, 
                                           value=st.session_state["annual_salary"], 
                                           step=500.0,
                                           key="salary_input")
            st.session_state["annual_salary"] = annual_salary
            st.session_state["hourly_rate"] = 0.0
    
    with col3:
        hours_per_week = st.number_input("Hours worked per week", 
                                        min_value=0.0, 
                                        value=st.session_state["hours_per_week"], 
                                        step=1.0,
                                        key="hours_input")
        st.session_state["hours_per_week"] = hours_per_week

    st.markdown("---")
    
    # Budget categories
    st.subheader("2️⃣ Monthly Budget Categories")
    st.caption("Add, edit, or remove categories. Blank rows are ignored.")
    
    col_needs, col_wants = st.columns(2)
    
    with col_needs:
        st.markdown("### 🏠 Needs (essentials)")
        needs_edit = st.data_editor(
            st.session_state["needs_df"],
            num_rows="dynamic",
            use_container_width=True,
            key="needs_editor",
            column_config={
                "Category": st.column_config.TextColumn(required=True),
                "Monthly $": st.column_config.NumberColumn(min_value=0.0, step=5.0, format="$%.2f"),
            },
        )
        st.session_state["needs_df"] = clean_budget_df(needs_edit)
        
        needs_total = st.session_state["needs_df"]["Monthly $"].sum()
        st.metric("Total Needs", f"${needs_total:,.2f}")
    
    with col_wants:
        st.markdown("### 🎯 Wants (discretionary)")
        wants_edit = st.data_editor(
            st.session_state["wants_df"],
            num_rows="dynamic",
            use_container_width=True,
            key="wants_editor",
            column_config={
                "Category": st.column_config.TextColumn(required=True),
                "Monthly $": st.column_config.NumberColumn(min_value=0.0, step=5.0, format="$%.2f"),
            },
        )
        st.session_state["wants_df"] = clean_budget_df(wants_edit)
        
        wants_total = st.session_state["wants_df"]["Monthly $"].sum()
        st.metric("Total Wants", f"${wants_total:,.2f}")

    st.markdown("---")
    
    # Goal settings
    st.subheader("3️⃣ Savings Goal (Optional)")
    
    col_goal1, col_goal2 = st.columns([2, 1])
    
    with col_goal1:
        goal_name = st.text_input("Goal name", 
                                 value=st.session_state["goal_name"],
                                 placeholder="e.g., Emergency fund, Vacation, New laptop",
                                 key="goal_name_input")
        st.session_state["goal_name"] = goal_name
    
    with col_goal2:
        goal_cost = st.number_input("Goal cost ($)", 
                                   min_value=0.0, 
                                   value=st.session_state["goal_cost"], 
                                   step=25.0,
                                   key="goal_cost_input")
        st.session_state["goal_cost"] = goal_cost

    st.markdown("---")
    
    # AI settings
    with st.expander("⚙️ Advanced: AI Settings"):
        st.caption("OpenAI API key should be set as environment variable: `export OPENAI_API_KEY=your_key`")
        model = st.text_input("OpenAI model", 
                             value=st.session_state["ai_model"],
                             help="Default: gpt-4o",
                             key="model_input")
        st.session_state["ai_model"] = model
        
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key:
            st.success("✅ API key detected")
        else:
            st.info("ℹ️ No API key found - will use fallback analysis")

    st.markdown("---")
    
    # Validation and next steps
    is_valid, message = validate_setup()
    
    if is_valid:
        st.success(message)
        st.session_state["setup_complete"] = True
        st.info("✨ Setup complete! Click on the **Dashboard** tab to see your opportunity cost analysis.")
    else:
        st.warning(message)
        st.session_state["setup_complete"] = False


def render_dashboard_tab(base: Dict, hr: float, work_hours: float, top3_df: pd.DataFrame, 
                         goal_name: str, goal_cost: float, analysis_text: str):
    """Render the dashboard overview tab"""
    
    if not st.session_state.get("setup_complete", False):
        st.warning("⚠️ Please complete the setup in the **Setup** tab first.")
        return
    
    st.header("📊 Opportunity Cost Dashboard")
    st.markdown("---")
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Monthly Income", f"${base['income']:,.0f}")
    with col2:
        st.metric("Effective Hourly Rate", f"${hr:,.2f}/hr")
    with col3:
        st.metric("Monthly Work Hours", f"{work_hours:,.1f} hrs")
    with col4:
        leftover_delta = base['leftover']
        st.metric("Leftover (Savings)", f"${base['leftover']:,.0f}", 
                 delta=None,
                 delta_color="normal" if leftover_delta >= 0 else "inverse")
    
    st.markdown("---")
    
    # Main visualization area
    col_chart, col_breakdown = st.columns([3, 2])
    
    with col_chart:
        st.subheader("Time Budget Breakdown")
        
        needs_hours = hours_equivalent(base["needs"], hr)
        wants_hours = hours_equivalent(base["wants"], hr)
        unallocated_hours = max(0.0, work_hours - needs_hours - wants_hours)
        
        chart_df = pd.DataFrame([
            {"Bucket": "Needs", "Hours": needs_hours, "Amount": base["needs"]},
            {"Bucket": "Wants", "Hours": wants_hours, "Amount": base["wants"]},
            {"Bucket": "Unallocated", "Hours": unallocated_hours, "Amount": base["leftover"]},
        ])
        
        chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("Hours:Q", title="Hours of work per month"),
                y=alt.Y("Bucket:N", title="", sort=["Needs", "Wants", "Unallocated"]),
                color=alt.Color("Bucket:N", 
                              scale=alt.Scale(domain=["Needs", "Wants", "Unallocated"],
                                            range=["#FF6B6B", "#4ECDC4", "#95E1D3"]),
                              legend=None),
                tooltip=["Bucket", 
                        alt.Tooltip("Hours:Q", format=",.1f", title="Hours"),
                        alt.Tooltip("Amount:Q", format="$,.0f", title="Monthly $")],
            )
            .properties(height=250)
        )
        st.altair_chart(chart, use_container_width=True)
        
        # Percentage breakdown
        st.caption(f"**Needs:** {(base['needs']/base['income']*100):.1f}% | "
                  f"**Wants:** {(base['wants']/base['income']*100):.1f}% | "
                  f"**Leftover:** {(base['leftover']/base['income']*100):.1f}%")
    
    with col_breakdown:
        st.subheader("Top Time Drains")
        st.dataframe(top3_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Goal tracking
        if goal_name and goal_cost > 0:
            st.subheader("🎯 Goal Tracker")
            gm = goal_metrics(goal_cost, hr, base["leftover"])
            
            st.write(f"**{goal_name}**")
            st.write(f"💰 Cost: ${goal_cost:,.0f}")
            st.write(f"⏱️ Time cost: **{gm['goal_hours']:.1f} hours** of work")
            
            if gm["weeks_to_goal"] != float("inf"):
                st.write(f"📅 Time to goal: **{gm['weeks_to_goal']:.1f} weeks**")
                st.caption(f"({gm['months_to_goal']:.1f} months)")
            else:
                st.error("⚠️ Not enough leftover to reach goal")
    
    st.markdown("---")
    
    # AI Summary
    st.subheader("🤖 AI Analysis Summary")
    st.info(analysis_text)


def render_insights_tab(base: Dict, hr: float, breakdown: pd.DataFrame, ai_bullets: List[str]):
    """Render the detailed insights tab"""
    
    if not st.session_state.get("setup_complete", False):
        st.warning("⚠️ Please complete the setup in the **Setup** tab first.")
        return
    
    st.header("💡 Detailed Insights")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Key Takeaways")
        
        if ai_bullets and len(ai_bullets) > 0:
            st.markdown("**AI-Generated Insights:**")
            for b in ai_bullets[:3]:
                st.markdown(f"• {b}")
        else:
            st.markdown("**Time Cost Analysis:**")
            st.write("• High-hour categories are the main opportunity cost drivers.")
            st.write("• Small changes here typically produce the largest time savings.")
            st.write("• Optimize these before focusing on small items.")
        
        st.markdown("---")
        
        st.subheader("Opportunity Cost View")
        needs_hours = hours_equivalent(base["needs"], hr)
        wants_hours = hours_equivalent(base["wants"], hr)
        
        st.metric("Needs (work hours/month)", f"{needs_hours:,.1f} hrs")
        st.metric("Wants (work hours/month)", f"{wants_hours:,.1f} hrs")
        
        total_hours = needs_hours + wants_hours
        st.caption(f"Total: {total_hours:,.1f} hours working to cover expenses")
    
    with col2:
        st.subheader("All Categories Breakdown")
        
        df_show = breakdown.copy()
        df_show["Share of income"] = (df_show["Share of income"] * 100).round(1).astype(str) + "%"
        df_show["Monthly $"] = df_show["Monthly $"].round(2)
        df_show["Hours of work"] = df_show["Hours of work"].round(1)
        
        st.dataframe(df_show, use_container_width=True, hide_index=True, height=400)
        
        # Summary stats
        st.markdown("---")
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            total_categories = len(breakdown)
            st.metric("Total Categories", total_categories)
        
        with col_b:
            avg_hours = breakdown["Hours of work"].mean()
            st.metric("Avg Hours/Category", f"{avg_hours:.1f}")
        
        with col_c:
            max_category = breakdown.loc[breakdown["Hours of work"].idxmax(), "Category"]
            st.metric("Biggest Drain", max_category)


def render_scenarios_tab(pay_type: str, hourly_rate: float, annual_salary: float, 
                        hours_per_week: float, needs_df: pd.DataFrame, wants_df: pd.DataFrame,
                        scenarios_out: List[Dict], base: Dict, hr: float):
    """Render the scenarios comparison tab"""
    
    if not st.session_state.get("setup_complete", False):
        st.warning("⚠️ Please complete the setup in the **Setup** tab first.")
        return
    
    st.header("🔮 What-If Scenarios")
    st.markdown("---")
    
    st.markdown("Explore how different changes to your budget or work schedule could improve your financial situation.")
    
    if not scenarios_out or len(scenarios_out) == 0:
        st.info("💡 Add some Want categories in the Setup tab to generate personalized scenarios.")
        return
    
    # Scenario selection
    st.subheader("Select a Scenario to Explore")
    
    titles = [s["title"] for s in scenarios_out]
    choice = st.radio("Available scenarios:", titles, index=0, horizontal=False)
    picked = next(s for s in scenarios_out if s["title"] == choice)
    
    st.markdown("---")
    
    # Apply scenario
    hr2, sal2, hpw2, needs2, wants2 = apply_scenario(
        pay_type, hourly_rate, annual_salary, hours_per_week, needs_df, wants_df, picked
    )
    
    income2 = monthly_income(pay_type, hr2, sal2, hpw2)
    eff_hr2 = effective_hourly_rate(pay_type, hr2, sal2, hpw2)
    needs2_total = float(needs2["Monthly $"].sum()) if len(needs2) > 0 else 0.0
    wants2_total = float(wants2["Monthly $"].sum()) if len(wants2) > 0 else 0.0
    updated = sums(income2, needs2_total, wants2_total)
    
    # Calculate deltas
    delta_leftover = updated["leftover"] - base["leftover"]
    delta_wants_hours = hours_equivalent(updated["wants"], eff_hr2) - hours_equivalent(base["wants"], hr)
    delta_income = updated["income"] - base["income"]
    
    # Impact metrics
    st.subheader("📈 Impact Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Change in Leftover", 
                 f"${delta_leftover:,.0f}",
                 delta=f"${delta_leftover:,.0f}",
                 delta_color="normal" if delta_leftover >= 0 else "inverse")
    
    with col2:
        st.metric("Change in Want Hours", 
                 f"{delta_wants_hours:,.1f} hrs",
                 delta=f"{delta_wants_hours:,.1f} hrs",
                 delta_color="inverse" if delta_wants_hours < 0 else "normal")
    
    with col3:
        st.metric("Change in Income", 
                 f"${delta_income:,.0f}",
                 delta=f"${delta_income:,.0f}",
                 delta_color="normal" if delta_income >= 0 else "inverse")
    
    with col4:
        pct_change = (delta_leftover / base["income"] * 100) if base["income"] > 0 else 0
        st.metric("% Impact on Income", 
                 f"{abs(pct_change):.1f}%",
                 delta=f"{pct_change:+.1f}%",
                 delta_color="normal" if pct_change >= 0 else "inverse")
    
    st.markdown("---")
    
    # Rationale
    col_rat, col_comp = st.columns([1, 2])
    
    with col_rat:
        st.subheader("Why This Scenario?")
        rationale = picked.get("ai_rationale") or picked.get("why") or "Scenario rationale unavailable."
        st.info(rationale)
        
        # Additional context
        st.markdown("**Scenario Details:**")
        scenario_type = picked.get("type", "")
        if scenario_type == "add_hours_per_week":
            st.write(f"• Adds {picked.get('value', 0):.0f} hours per week")
        elif scenario_type == "cut_total_wants_pct":
            st.write(f"• Reduces all wants by {picked.get('value', 0)*100:.0f}%")
        elif scenario_type == "cut_one_want_pct":
            st.write(f"• Reduces {picked.get('category', 'category')} by {picked.get('value', 0)*100:.0f}%")
        elif scenario_type == "delta_wants_total":
            st.write(f"• Adjusts wants by ${picked.get('value', 0):,.0f}")
    
    with col_comp:
        st.subheader("Before vs. After Comparison")
        
        # Comparison chart
        comp_df = pd.DataFrame([
            {"Scenario": "Current", "Needs": hours_equivalent(base["needs"], hr), 
             "Wants": hours_equivalent(base["wants"], hr)},
            {"Scenario": "With Change", "Needs": hours_equivalent(updated["needs"], eff_hr2), 
             "Wants": hours_equivalent(updated["wants"], eff_hr2)},
        ]).melt(id_vars=["Scenario"], var_name="Category", value_name="Hours")
        
        comp_chart = (
            alt.Chart(comp_df)
            .mark_bar()
            .encode(
                x=alt.X("Hours:Q", title="Hours of work per month"),
                y=alt.Y("Scenario:N", title=""),
                color=alt.Color("Category:N", 
                              scale=alt.Scale(domain=["Needs", "Wants"],
                                            range=["#FF6B6B", "#4ECDC4"])),
                tooltip=["Scenario", "Category", alt.Tooltip("Hours:Q", format=",.1f")],
            )
            .properties(height=200)
        )
        st.altair_chart(comp_chart, use_container_width=True)
        
        # Detailed comparison table
        comp_table = pd.DataFrame([
            {"Metric": "Monthly Income", "Current": f"${base['income']:,.0f}", "With Change": f"${updated['income']:,.0f}"},
            {"Metric": "Needs", "Current": f"${base['needs']:,.0f}", "With Change": f"${updated['needs']:,.0f}"},
            {"Metric": "Wants", "Current": f"${base['wants']:,.0f}", "With Change": f"${updated['wants']:,.0f}"},
            {"Metric": "Leftover", "Current": f"${base['leftover']:,.0f}", "With Change": f"${updated['leftover']:,.0f}"},
        ])
        st.dataframe(comp_table, use_container_width=True, hide_index=True)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application entry point"""
    
    # Page config
    st.set_page_config(
        page_title="Opportunity Cost Visualizer",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Initialize state
    init_state()
    
    # Header
    st.title("💰 Opportunity Cost Visualizer")
    st.markdown("**Transform your budget into time** — see how many hours of work each expense really costs you.")
    
    # Progress indicator
    is_valid, _ = validate_setup()
    if is_valid:
        st.success("✅ Setup complete — explore your insights below!")
    else:
        st.info("👉 Start by completing the setup in the first tab")
    
    st.markdown("---")
    
    # Tabs
    tab_setup, tab_dash, tab_insights, tab_scenarios = st.tabs([
        "📝 Setup", 
        "📊 Dashboard", 
        "💡 Insights", 
        "🔮 Scenarios"
    ])
    
    # Get current data from session state
    needs_df = st.session_state["needs_df"]
    wants_df = st.session_state["wants_df"]
    pay_type = st.session_state["pay_type"]
    hourly_rate = st.session_state["hourly_rate"]
    annual_salary = st.session_state["annual_salary"]
    hours_per_week = st.session_state["hours_per_week"]
    goal_name = st.session_state["goal_name"]
    goal_cost = st.session_state["goal_cost"]
    model = st.session_state["ai_model"]
    
    # Compute baseline metrics
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
    
    # Only run AI insights if setup is complete (to avoid wasting API calls)
    if is_valid:
        analysis_text, scenarios_out = ai_insights(model=model, snapshot=snapshot, 
                                                   top3=top3_records, scenarios=scenarios)
    else:
        analysis_text = "Complete the setup to see AI-powered insights."
        scenarios_out = scenarios
    
    ai_bullets = st.session_state.get("ai_time_drain_bullets", [])
    
    # Render tabs
    with tab_setup:
        render_setup_tab()
    
    with tab_dash:
        render_dashboard_tab(base, hr, work_hours, top3_df, goal_name, goal_cost, analysis_text)
    
    with tab_insights:
        render_insights_tab(base, hr, breakdown, ai_bullets)
    
    with tab_scenarios:
        render_scenarios_tab(pay_type, hourly_rate, annual_salary, hours_per_week, 
                           needs_df, wants_df, scenarios_out, base, hr)
    
    # Footer
    st.markdown("---")
    st.caption("💡 Tip: Hover over charts for details. Use the data editors to customize your budget.")


if __name__ == "__main__":
    main()
