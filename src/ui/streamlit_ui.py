import streamlit as st
import requests

# -------------------------
# CONFIG
# -------------------------
API_URL = "http://127.0.0.1:8000/api/v1"

st.set_page_config(
    page_title="💳 Credit Card Spend Analyzer",
    layout="wide"
)

# -------------------------
# SESSION STATE
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.title("🔀 Mode")
mode = st.sidebar.radio("Select Mode", ["User", "Admin"])

if st.sidebar.button("🔄 Reset Chat"):
    st.session_state.messages = []
    st.rerun()

# =========================
# 💬 USER MODE
# =========================
if mode == "User":
    st.title("💳 Credit Card Spend Analyzer")

    # -------------------------
    # CHAT HISTORY
    # -------------------------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):

            data = msg.get("data")

            if data:
                st.write("📌 QUERY:")
                st.write(data.get("query"))

                for ans in data.get("answers", []):
                    st.write("----")
                    st.write("🧠 SUB QUERY:")
                    st.write(ans.get("sub_query"))

                    st.write("💬 ANSWER:")
                    st.write(ans.get("answer"))

                    st.write("📚 RETRIEVED RESULTS:")

                    for r in ans.get("retrieved_results", []):
                        st.json(r)   # EXACT chunk structure
            else:
                st.write(msg["content"])

    user_input = st.chat_input("Ask your question...")

    if user_input:

        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json={"question": user_input},
                    timeout=180
                )

                if response.status_code == 200:
                    data = response.json()

                    # store RAW response (NO CHANGE)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "data": data
                    })

                    # render assistant
                    with st.chat_message("assistant"):

                        st.write("📌 QUERY:")
                        st.write(data.get("query"))

                        for ans in data.get("answers", []):
                            st.write("----")
                            st.write("🧠 SUB QUERY:")
                            st.write(ans.get("sub_query"))

                            st.write("💬 ANSWER:")
                            st.write(ans.get("answer"))

                            st.write("📚 RETRIEVED RESULTS:")

                            for r in ans.get("retrieved_results", []):
                                st.json(r)   # EXACT SAME JSON STRUCTURE

                else:
                    st.error(f"❌ Backend error: {response.text}")

            except Exception as e:
                st.error(f"⚠️ Connection error: {e}")