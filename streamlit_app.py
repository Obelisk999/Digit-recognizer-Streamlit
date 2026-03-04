import streamlit as st

# --- Page Config ---
st.set_page_config(
    page_title="LaunchPad",
    page_icon="🚀",
    layout="centered",
)

# --- Custom CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

.stApp {
    background: #0a0a0f;
    color: #f0ede8;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em;
}

.hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
    border-bottom: 1px solid #1e1e2e;
    margin-bottom: 2rem;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.5rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    line-height: 1.05;
    background: linear-gradient(135deg, #e8ff4d 0%, #ff6b35 50%, #ff3cac 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}

.hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #666680;
    margin-top: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.metric-card {
    background: #111118;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.2s;
}

.metric-card:hover {
    border-color: #e8ff4d44;
}

.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #555570;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.35rem;
}

.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #e8ff4d;
    letter-spacing: -0.03em;
    line-height: 1;
}

.metric-delta {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #4dff91;
    margin-top: 0.3rem;
}

.tag {
    display: inline-block;
    background: #1a1a28;
    border: 1px solid #2a2a40;
    border-radius: 999px;
    padding: 0.2rem 0.75rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #8888aa;
    margin-right: 0.4rem;
    margin-bottom: 0.4rem;
    letter-spacing: 0.05em;
}

.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #444460;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e1e2e;
}

div[data-testid="stButton"] button {
    background: #e8ff4d !important;
    color: #0a0a0f !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.03em !important;
    padding: 0.6rem 1.5rem !important;
    transition: opacity 0.2s !important;
}

div[data-testid="stButton"] button:hover {
    opacity: 0.85 !important;
}

div[data-testid="stTextInput"] input,
div[data-testid="stSelectbox"] select,
div[data-testid="stTextArea"] textarea {
    background: #111118 !important;
    border: 1px solid #1e1e2e !important;
    color: #f0ede8 !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
}

div[data-testid="stSlider"] .stSlider {
    accent-color: #e8ff4d;
}

[data-testid="stSidebar"] {
    background: #08080d !important;
    border-right: 1px solid #1e1e2e;
}

.footer {
    text-align: center;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #333348;
    padding: 2rem 0 1rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border-top: 1px solid #1a1a28;
    margin-top: 3rem;
}
</style>
""", unsafe_allow_html=True)


# --- Session State ---
if "ideas" not in st.session_state:
    st.session_state.ideas = []
if "submitted" not in st.session_state:
    st.session_state.submitted = False


# --- Hero ---
st.markdown("""
<div class="hero">
    <div class="hero-title">LAUNCHPAD</div>
    <div class="hero-sub">Your startup command center</div>
</div>
""", unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    st.markdown('<div class="section-label">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("", ["📊 Dashboard", "💡 Idea Tracker", "⚙️ Settings"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div class="section-label">Status</div>', unsafe_allow_html=True)
    stage = st.selectbox("Stage", ["Ideation", "MVP", "Seed", "Series A", "Growth"])
    runway = st.slider("Runway (months)", 0, 36, 18)

    if runway < 6:
        st.warning("⚠️ Low runway!")
    elif runway > 18:
        st.success("✅ Healthy runway")


# ─── DASHBOARD ───────────────────────────────────────────────────────────────
if page == "📊 Dashboard":

    st.markdown('<div class="section-label">Key Metrics</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">MRR</div>
            <div class="metric-value">$4.2K</div>
            <div class="metric-delta">↑ 18% this month</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Users</div>
            <div class="metric-value">1,340</div>
            <div class="metric-delta">↑ 230 new</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Churn</div>
            <div class="metric-value">2.1%</div>
            <div class="metric-delta">↓ 0.4% lower</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Growth Chart</div>', unsafe_allow_html=True)

    import pandas as pd
    import numpy as np

    months = ["Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar"]
    mrr = [800, 1200, 1600, 2100, 2800, 3400, 3900, 4200]
    users = [120, 210, 380, 540, 780, 1020, 1180, 1340]

    chart_data = pd.DataFrame({"MRR ($)": mrr, "Users": users}, index=months)
    st.line_chart(chart_data["MRR ($)"], use_container_width=True)

    st.markdown('<div class="section-label">Current Stage</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <span class="tag">🏁 {stage}</span>
    <span class="tag">⏳ {runway} mo runway</span>
    <span class="tag">🌍 SaaS</span>
    <span class="tag">🤖 AI-powered</span>
    """, unsafe_allow_html=True)


# ─── IDEA TRACKER ─────────────────────────────────────────────────────────────
elif page == "💡 Idea Tracker":

    st.markdown('<div class="section-label">Capture Ideas</div>', unsafe_allow_html=True)

    idea_text = st.text_area("New idea", placeholder="What's the next big thing?", height=100)
    priority = st.selectbox("Priority", ["🔥 High", "⚡ Medium", "🧊 Low"])

    if st.button("Add Idea"):
        if idea_text.strip():
            st.session_state.ideas.append({"idea": idea_text.strip(), "priority": priority})
            st.success("Idea logged.")
        else:
            st.error("Write something first.")

    if st.session_state.ideas:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Backlog</div>', unsafe_allow_html=True)
        for i, item in enumerate(reversed(st.session_state.ideas)):
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{item['priority']}</div>
                <div style="font-family:'Syne',sans-serif;font-weight:600;font-size:1rem;color:#f0ede8;margin-top:0.3rem;">{item['idea']}</div>
            </div>
            """, unsafe_allow_html=True)

        if st.button("Clear All"):
            st.session_state.ideas = []
            st.rerun()
    else:
        st.markdown('<p style="font-family:\'Space Mono\',monospace;font-size:0.8rem;color:#444460;margin-top:1.5rem;">No ideas yet. Start brainstorming.</p>', unsafe_allow_html=True)


# ─── SETTINGS ─────────────────────────────────────────────────────────────────
elif page == "⚙️ Settings":

    st.markdown('<div class="section-label">Company Info</div>', unsafe_allow_html=True)

    name = st.text_input("Startup name", value="LaunchPad Inc.")
    tagline = st.text_input("Tagline", value="Move fast, build things.")
    industry = st.selectbox("Industry", ["SaaS", "FinTech", "HealthTech", "EdTech", "Climate", "Developer Tools", "Consumer", "Other"])
    team_size = st.slider("Team size", 1, 50, 4)

    if st.button("Save Settings"):
        st.success(f"Saved — {name} · {industry} · {team_size} people")


# --- Footer ---
st.markdown('<div class="footer">Built with Streamlit · LaunchPad v0.1</div>', unsafe_allow_html=True)
