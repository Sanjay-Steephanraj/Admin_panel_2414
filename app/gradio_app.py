import gradio as gr
import requests
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

API_URL = "http://127.0.0.1:8000/chat/query"  # Change if your FastAPI runs on different port


# ---------------------------------------------------
# BACKEND CALL
# ---------------------------------------------------

def ask_backend(question, session_id):
    if not question.strip():
        return "Please enter a valid question.", session_id or "", session_id or ""

    try:
        # Always send the current State value. It is empty only on the first
        # request; the response session_id becomes the next State value.
        payload = {
            "question": question,
            "session_id": session_id or "",
        }

        logger.info("Sending request with session_id=%r", session_id or "")

        response = requests.post(
            API_URL,
            json=payload,
            timeout=90,
        )
        response.raise_for_status()

        data = response.json()

        # Reuse the backend session_id for all subsequent messages.
        returned_session_id = data.get("session_id") or session_id or ""

        logger.info(
            "Received response with session_id=%r",
            returned_session_id,
        )

        return (
            data.get("response", "No response received from backend."),
            returned_session_id,
            returned_session_id,
        )

    except requests.exceptions.ConnectionError:
        return (
            "❌ Cannot connect to backend. Ensure FastAPI is running.",
            session_id or "",
            session_id or "",
        )

    except requests.exceptions.Timeout:
        return (
            "⏳ Backend is taking too long to respond.",
            session_id or "",
            session_id or "",
        )

    except requests.exceptions.HTTPError as e:
        return (
            f"⚠️ Backend returned error: {e}",
            session_id or "",
            session_id or "",
        )

    except Exception as e:
        return (
            f"Unexpected error: {str(e)}",
            session_id or "",
            session_id or "",
        )


# ---------------------------------------------------
# CLASSICAL WHITE THEME
# ---------------------------------------------------

custom_css = """
body {
    background-color: #ffffff !important;
    font-family: 'Georgia', serif;
}

.gradio-container {
    background-color: #ffffff !important;
    max-width: 900px !important;
    margin: auto !important;
}

#title {
    text-align: center;
    font-size: 34px;
    font-weight: 600;
    margin-top: 30px;
    color: #111111;
}

#subtitle {
    text-align: center;
    font-size: 16px;
    color: #555555;
    margin-bottom: 40px;
}

/* INPUT BOX */
textarea {
    background-color: #ffffff !important;
    color: #111111 !important;
    border-radius: 8px !important;
    border: 1px solid #cccccc !important;
    font-size: 16px !important;
}

/* BUTTON */
button {
    background-color: #111111 !important;
    color: white !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
}

/* OUTPUT BOX */
.output-box {
    background-color: black !important;
    color: black !important;
    border: 1px solid #dddddd !important;
    border-radius: 8px !important;
    padding: 25px !important;
    font-size: 16px !important;
    line-height: 1.7 !important;
}
"""


# ---------------------------------------------------
# UI LAYOUT
# ---------------------------------------------------

with gr.Blocks(css=custom_css) as app:

    # The backend's MemorySaver is keyed by this session id. Keep it in the
    # browser session so clarification replies reuse the same LangGraph thread.
    session_state = gr.State(value="")

    gr.Markdown("<div id='title'>Donor Intelligence Assistant</div>")
    gr.Markdown("<div id='subtitle'>Executive Insights on Donors, Campaigns & Giving Performance</div>")

    question_input = gr.Textbox(
        placeholder="Ask about donors, campaigns, revenue trends...",
        lines=3,
        label="Your Question"
    )

    submit_btn = gr.Button("Generate Insight")

    response_output = gr.Markdown(elem_classes="output-box")
    session_display = gr.Textbox(label="Session ID (diagnostic)", interactive=False)

    submit_btn.click(
        fn=ask_backend,
        inputs=[question_input, session_state],
        outputs=[response_output, session_state, session_display]
    )


# ---------------------------------------------------
# RUN APP
# ---------------------------------------------------

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )
