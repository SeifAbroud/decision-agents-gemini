"""Point d'entrée du dashboard (Streamlit Cloud : main file = app/streamlit_app.py)."""
import streamlit as st

st.set_page_config(page_title="Aide à la décision · Favorita", page_icon="📊", layout="wide")
nav = st.navigation([
    st.Page("views/home.py", title="Vue d'ensemble", icon="📊", default=True),
    st.Page("views/explorer.py", title="Magasins et familles", icon="🏬"),
    st.Page("views/forecast.py", title="Prévisions", icon="📈"),
    st.Page("views/agents.py", title="Agents et recommandations", icon="🤖"),
    st.Page("views/assistant.py", title="Assistant", icon="💬"),
    st.Page("views/method.py", title="Méthodologie", icon="🧭"),
])
nav.run()
