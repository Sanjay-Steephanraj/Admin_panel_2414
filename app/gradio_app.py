import gradio as gr
import requests

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

API_URL = "http://127.0.0.1:8084/query"  # Change if your FastAPI runs on different port


# ---------------------------------------------------
# BACKEND CALL
# ---------------------------------------------------

def ask_backend(question):

    if not question.strip():
        return "Please enter a valid question."

    try:
        response = requests.post(
            API_URL,
            json={"question": question},
            timeout=90
        )

        response.raise_for_status()

        data = response.json()
        return data.get("response", "No response received from backend.")

    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to backend. Ensure FastAPI is running."

    except requests.exceptions.Timeout:
        return "⏳ Backend is taking too long to respond."

    except requests.exceptions.HTTPError as e:
        return f"⚠️ Backend returned error: {e}"

    except Exception as e:
        return f"Unexpected error: {str(e)}"


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

    gr.Markdown("<div id='title'>Donor Intelligence Assistant</div>")
    gr.Markdown("<div id='subtitle'>Executive Insights on Donors, Campaigns & Giving Performance</div>")

    question_input = gr.Textbox(
        placeholder="Ask about donors, campaigns, revenue trends...",
        lines=3,
        label="Your Question"
    )

    submit_btn = gr.Button("Generate Insight")

    response_output = gr.Markdown(elem_classes="output-box")

    submit_btn.click(
        fn=ask_backend,
        inputs=question_input,
        outputs=response_output
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
