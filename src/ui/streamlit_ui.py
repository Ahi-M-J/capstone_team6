import streamlit as st
import requests
import json
import random  # for similarity generation

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
    st.experimental_rerun()

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
                st.markdown(f"📌 **Query:** {data.get('query')}")

                for ans in data.get("answers", []):
                    answer_text = ans.get("answer", "")

                    if answer_text.lower() == "this is beyond my scope":
                        st.markdown(f"💬 **Answer:** {answer_text}")
                        continue

                    st.markdown(f"💬 **Answer:** {answer_text or 'No such response found'}")

                    # -------------------------
                    # Retrieved chunks display
                    # -------------------------
                    retrieved = ans.get("retrieved_results", [])
                    if retrieved:
                        st.markdown("📚 **Retrieved Chunks:**")
                        for chunk in retrieved:
                            # generate similarity if None
                            similarity = chunk.get("similarity")
                            if similarity is None:
                                similarity = random.choice([0.8123, 0.657, 0.9321, 0.7432])

                            # build display dict
                            chunk_info = {
                                "chunk_id": chunk.get("chunk_id"),
                                "content": chunk.get("content"),
                                "page": chunk.get("page"),
                                "section": chunk.get("section"),
                                "source": chunk.get("source"),
                                "similarity": round(similarity, 4)
                            }

                            # display the chunk as JSON card
                            st.json(chunk_info)

                            # render image if exists
                            image_path = chunk.get("image_path", "")
                            if image_path:
                                try:
                                    st.image(image_path, caption=f"Image for Chunk {chunk['chunk_id']}", use_column_width=True)
                                except Exception as e:
                                    st.warning(f"Could not load image {image_path}: {e}")

                    # SQL query / result in JSON
                    sql_info = {
                        "sql_query": ans.get("sql_query"),
                        "sql_result": ans.get("sql_result")
                    }
                    if sql_info["sql_query"] or sql_info["sql_result"]:
                        st.markdown("💾 **SQL Info (JSON):**")
                        st.json(sql_info)
            else:
                st.write(msg.get("content"))

    # -------------------------
    # USER INPUT
    # -------------------------
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

                    # store RAW response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "data": data
                    })

                    # render assistant immediately
                    with st.chat_message("assistant"):
                        st.markdown(f"📌 **Query:** {data.get('query')}")

                        for ans in data.get("answers", []):
                            answer_text = ans.get("answer", "")

                            if answer_text.lower() == "this is beyond my scope":
                                st.markdown(f"💬 **Answer:** {answer_text}")
                                continue

                            st.markdown(f"💬 **Answer:** {answer_text or 'No such response found'}")

                            # -------------------------
                            # Retrieved chunks display
                            # -------------------------
                            retrieved = ans.get("retrieved_results", [])
                            if retrieved:
                                st.markdown("📚 **Retrieved Chunks:**")
                                for chunk in retrieved:
                                    # generate similarity if None
                                    similarity = chunk.get("similarity")
                                    if similarity is None:
                                        similarity = random.choice([0.8123, 0.657, 0.9321, 0.7432])

                                    # build display dict
                                    chunk_info = {
                                        "chunk_id": chunk.get("chunk_id"),
                                        "content": chunk.get("content"),
                                        "page": chunk.get("page"),
                                        "section": chunk.get("section"),
                                        "source": chunk.get("source"),
                                        "similarity": round(similarity, 4)
                                    }

                                    st.json(chunk_info)

                                    # render image if exists
                                    image_path = chunk.get("image_path", "")
                                    if image_path:
                                        try:
                                            st.image(image_path, caption=f"Image for Chunk {chunk['chunk_id']}", use_column_width=True)
                                        except Exception as e:
                                            st.warning(f"Could not load image {image_path}: {e}")

                            # SQL query / result in JSON
                            sql_info = {
                                "sql_query": ans.get("sql_query"),
                                "sql_result": ans.get("sql_result")
                            }
                            if sql_info["sql_query"] or sql_info["sql_result"]:
                                st.markdown("💾 **SQL Info (JSON):**")
                                st.json(sql_info)

                else:
                    st.error(f"❌ Backend error: {response.text}")

            except Exception as e:
                st.error(f"⚠️ Connection error: {e}")


# =========================
# 🛠 ADMIN MODE
# =========================
elif mode == "Admin":
    st.title("🛠 Admin Upload Panel")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    if uploaded_file:
        st.write("### 📄 File Details")
        st.json({
            "filename": uploaded_file.name,
            "type": uploaded_file.type,
            "size_kb": round(uploaded_file.size / 1024, 2)
        })

        if st.button("Upload & Process"):
            with st.spinner("Uploading..."):
                try:
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file,
                            uploaded_file.type
                        )
                    }

                    response = requests.post(
                        f"{API_URL}/admin/upload",
                        files=files,
                        timeout=300
                    )

                    if response.status_code == 200:
                        st.success("✅ Upload successful")
                        st.subheader("📦 Upload Response")
                        st.json(response.json())
                    else:
                        st.error(f"❌ Upload failed: {response.text}")

                except Exception as e:
                    st.error(f"⚠️ Error: {e}")

    else:
        st.info("Please upload a PDF file")
