"""
Legal Document Assistant — Streamlit UI.

Upload a PDF, get a summary, and ask questions about it.
Run with:
    streamlit run streamlit_app.py
"""

import os
import tempfile
import traceback

import streamlit as st

from pipeline import LegalDocumentPipeline


# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="Legal Document Assistant",
    layout="wide",
)

st.title("Legal Document Assistant")
st.caption(
    "Upload a legal document to get a summary and ask questions about it."
)


# =========================================================
# Session state — load pipeline once
# =========================================================
if "pipeline" not in st.session_state:
    with st.spinner("Loading models (first run only)..."):
        st.session_state.pipeline = LegalDocumentPipeline()
    st.session_state.doc_result    = None
    st.session_state.chat_history  = []
    st.session_state.doc_processed = False
    st.session_state.temp_path     = None


# =========================================================
# Sidebar — upload + metadata
# =========================================================
with st.sidebar:
    st.header("Upload Document")

    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file is not None:
        # Save upload to a persistent temp location across reruns.
        tmp_dir = tempfile.gettempdir()
        temp_path = os.path.join(tmp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        st.session_state.temp_path = temp_path

        if st.button("Process Document", use_container_width=True):
            with st.spinner("Processing document — extracting, "
                            "chunking, and summarizing..."):
                try:
                    result = st.session_state.pipeline.process_document(
                        st.session_state.temp_path
                    )
                    st.session_state.doc_result    = result
                    st.session_state.doc_processed = True
                    st.session_state.chat_history  = []
                    st.success("Document processed successfully")
                except Exception as exc:  # pylint: disable=broad-except
                    st.error(f"Error processing document: {exc}")
                    st.code(traceback.format_exc())

    # -----------------------------------------------------
    if st.session_state.doc_processed:
        st.divider()
        st.subheader("Document Info")

        r = st.session_state.doc_result
        st.write(f"**Type:** {r['doc_type'].replace('_', ' ').title()}")
        st.write(f"**Sections:** {r['total_chunks']}")
        if r.get("parties"):
            st.write(f"**Parties:** {', '.join(r['parties'][:3])}")
        if r.get("dates"):
            st.write(f"**Date:** {r['dates'][0]}")
        if r.get("jurisdiction") and r["jurisdiction"] != "Unknown":
            st.write(f"**Jurisdiction:** {r['jurisdiction']}")

        if st.button("Reset", use_container_width=True):
            st.session_state.pipeline.reset()
            st.session_state.doc_processed = False
            st.session_state.doc_result    = None
            st.session_state.chat_history  = []
            st.session_state.temp_path     = None
            st.rerun()


# =========================================================
# Main area — two tabs
# =========================================================
tab_summary, tab_qa = st.tabs(["Summary", "Ask Questions"])


# ----- Tab 1: Summary -----------------------------------
with tab_summary:
    if not st.session_state.doc_processed:
        st.info("Upload and process a document to see its summary.")
    else:
        r = st.session_state.doc_result
        st.subheader("Document Summary")
        st.write(r["summary"])
        st.caption(
            f"Summary length: {r['word_count']} words  |  "
            f"Sections processed: {r['total_chunks']}"
        )


# ----- Tab 2: QA chat -----------------------------------
with tab_qa:
    if not st.session_state.doc_processed:
        st.info("Upload and process a document to ask questions.")
    else:
        st.subheader("Ask Questions About This Document")

        # replay conversation history
        for question, answer_dict in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                st.write(answer_dict["plain_answer"])
                if answer_dict.get("found"):
                    st.caption(
                        f"{answer_dict['source_display']}  |  "
                        f"Confidence: {answer_dict['confidence']}"
                    )
                    with st.expander("View exact text from document"):
                        st.write(answer_dict["raw_span"])

        # input box
        question = st.chat_input("Ask a question about the document...")

        if question:
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("Finding answer..."):
                    try:
                        response = st.session_state.pipeline.answer(question)
                        st.write(response["plain_answer"])

                        if response.get("found"):
                            st.caption(
                                f"{response['source_display']}  |  "
                                f"Confidence: {response['confidence']}"
                            )
                            with st.expander(
                                "View exact text from document"
                            ):
                                st.write(response["raw_span"])

                        st.session_state.chat_history.append(
                            (question, response)
                        )
                    except Exception as exc:  # pylint: disable=broad-except
                        st.error(f"Error getting answer: {exc}")
                        st.code(traceback.format_exc())
