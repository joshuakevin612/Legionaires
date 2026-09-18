"""
Krafters Link — a central hub for CodeKrafters links & tasks.

Run with:
    streamlit run app.py
"""

import streamlit as st

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Krafters Link",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------
# EDIT ME — your links live here. Add / remove / reorder freely.
# `icon` accepts any emoji; swap in a real photo later by changing the
# card HTML below to an <img src="..."> if you'd rather use images.
# --------------------------------------------------------------------------
LINKS = [
    {
        "icon": "🚀",
        "title": "Launchpad 3.0 Website",
        "tag": "PROJECT LINK",
        "description": "Official website for Launchpad 3.0 — CodeKrafters' flagship annual tech fest.",
        "url": "#",
    },
    {
        "icon": "💻",
        "title": "Web Development Task",
        "tag": "DEVELOPMENT TASK",
        "description": "Interactive web development challenge and guidelines for Krafters.",
        "url": "#",
    },
    {
        "icon": "🌐",
        "title": "Web3-Den",
        "tag": "WEB3 DOMAIN",
        "description": "Showcase of our Web3 domain projects and innovations.",
        "url": "#",
    },
    {
        "icon": "🎨",
        "title": "Design Guild",
        "tag": "DESIGN TEAM",
        "description": "Brand assets, Figma files, and design task board for the creative wing.",
        "url": "#",
    },
    {
        "icon": "📢",
        "title": "Announcements",
        "tag": "COMMUNITY",
        "description": "Latest updates, event dates, and community shout-outs.",
        "url": "#",
    },
    {
        "icon": "🏆",
        "title": "Hackathon Hub",
        "tag": "EVENT",
        "description": "Rules, timelines, and submission portal for the next Krafters hackathon.",
        "url": "#",
    },
]

# --------------------------------------------------------------------------
# Styling — warm cream / amber-yellow palette, bold black type, pill chrome
# --------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700;800;900&family=Inter:wght@400;500;600&display=swap');

:root {
    --bg: #f7e8a8;
    --bg-soft: #f3e096;
    --card: #fdf6d8;
    --ink: #14110c;
    --amber: #f2a90c;
    --amber-dim: #d99a0a;
    --muted: #6b6350;
    --border: #14110c;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: radial-gradient(circle at 15% 0%, var(--bg-soft) 0%, var(--bg) 55%);
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding-top: 2.2rem;
    max-width: 1200px;
}

/* ---- top bar ---- */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 2.6rem;
}
.brand-pill {
    background: var(--ink);
    color: var(--bg);
    border-radius: 999px;
    padding: 12px 22px;
    font-weight: 700;
    font-size: 1.1rem;
    display: inline-flex;
    align-items: center;
    box-shadow: 3px 3px 0 0 var(--amber);
}
.brand {
    text-align: right;
}
.brand-title {
    font-family: 'Poppins', sans-serif;
    font-weight: 900;
    font-size: 2.6rem;
    color: var(--ink);
    letter-spacing: -0.01em;
    line-height: 1;
}
.brand-title .accent {
    color: var(--amber);
    position: relative;
}
.brand-title .accent::after {
    content: "";
    position: absolute;
    left: 2px; right: 2px; bottom: 4px;
    height: 5px;
    background: var(--ink);
}
.brand-sub {
    color: var(--muted);
    font-size: 0.92rem;
    font-weight: 500;
    margin-top: 4px;
}

/* ---- section heading ---- */
.section-heading {
    text-align: center;
    margin-bottom: 2.4rem;
}
.section-heading h2 {
    font-family: 'Poppins', sans-serif;
    font-weight: 800;
    font-size: 2.1rem;
    color: var(--ink);
    display: inline-block;
    position: relative;
    letter-spacing: 0.02em;
}
.section-heading h2::after {
    content: "";
    position: absolute;
    left: 50%;
    bottom: -8px;
    width: 50px;
    height: 5px;
    background: var(--ink);
    transform: translateX(-50%);
    border-radius: 3px;
}

/* ---- cards ---- */
div[data-testid="stVerticalBlockBorderWrapper"]:has(.card-marker) {
    background: var(--card);
    border: 2.5px solid var(--border);
    border-radius: 22px;
    padding: 6px;
    box-shadow: 7px 7px 0 0 var(--ink);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.card-marker):hover {
    transform: translate(-3px, -3px);
    box-shadow: 10px 10px 0 0 var(--amber);
}

.card-avatar {
    width: 92px;
    height: 92px;
    border-radius: 50%;
    background: linear-gradient(145deg, var(--amber) 0%, var(--amber-dim) 100%);
    border: 2.5px solid var(--ink);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2.4rem;
    margin: 6px auto 14px auto;
}
.card-title {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 1.25rem;
    color: var(--ink);
    text-align: center;
    margin-bottom: 2px;
}
.card-tag {
    text-align: center;
    color: var(--amber-dim);
    font-weight: 700;
    font-size: 0.78rem;
    letter-spacing: 0.06em;
    margin-bottom: 10px;
}
.card-desc {
    text-align: center;
    color: var(--muted);
    font-size: 0.9rem;
    line-height: 1.45;
    min-height: 60px;
    padding: 0 6px;
}

/* ---- open-link pill button ---- */
div.stLinkButton { display: flex; justify-content: center; margin: 14px 0 10px 0; }
div.stLinkButton > a {
    background: var(--ink) !important;
    color: var(--bg) !important;
    border-radius: 999px !important;
    padding: 10px 26px !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.03em;
    border: none !important;
    box-shadow: 3px 3px 0 0 var(--amber);
    transition: transform 0.15s ease;
}
div.stLinkButton > a:hover {
    transform: translateY(-2px);
    color: var(--amber) !important;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Top bar
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="topbar">
        <div class="brand-pill">⌁ CK</div>
        <div class="brand">
            <div class="brand-title">KRAFTERS <span class="accent">LINK</span></div>
            <div class="brand-sub">The central hub for CodeKrafters links &amp; tasks</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-heading"><h2>ALL LINKS</h2></div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Link cards, 3 per row
# --------------------------------------------------------------------------
cols_per_row = 3
for row_start in range(0, len(LINKS), cols_per_row):
    row_items = LINKS[row_start : row_start + cols_per_row]
    cols = st.columns(cols_per_row)
    for col, item in zip(cols, row_items):
        with col:
            with st.container(border=True):
                st.markdown('<span class="card-marker"></span>', unsafe_allow_html=True)
                st.markdown(f'<div class="card-avatar">{item["icon"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="card-title">{item["title"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="card-tag">{item["tag"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="card-desc">{item["description"]}</div>', unsafe_allow_html=True)
                st.link_button("OPEN LINK ↗", item["url"], use_container_width=True)
