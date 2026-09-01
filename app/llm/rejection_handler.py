import logging
from llm.gloo_client import get_llm
from langchain_core.messages import SystemMessage, HumanMessage
from components.prompt_config import get_error_message

logger = logging.getLogger(__name__)

def generate_dynamic_error_reply(question: str, error_type: str) -> str:
    """
    Generates a dynamic, conversational error message based on the error type.
    Paraphrases the static error message so it does not sound repetitive.
    """
    static_message = get_error_message(error_type)
    
    # Fast path for DB failures/security just return static so we don't risk looping/latency on critical errors
    if error_type in ["db_failure", "security", "injection"]:
        return static_message

    # Mask PII before sending to LLM. Fail-closed: if masking raises, return the
    # static error message (the existing non-LLM fast path) — never send raw.
    try:
        from data_masking import masker
        masked_question, mapping = masker.mask_text(question)
    except RuntimeError:
        logger.warning("PII masking failed — returning static error message")
        return static_message

    try:
        system_prompt = (
            "You are the Donor Portal CRM Assistant, helping an admin. "
            "The user asked a question we cannot answer for the following reason:\n"
            f"REASON: {static_message}\n\n"
            "Your task: Politely respond to the admin. "
            "CRITICAL INSTRUCTION: You must drastically vary your sentence structure, vocabulary, length, and flow "
            "so it never sounds like a canned response. Acknowledge their exact question briefly (if appropriate) "
            "before firmly pivoting to what you can actually help with (donor data, ministries, pipelines, or payments). "
            "Never copy the phrasing of the REASON above verbatim. Use maximum conversational variety. Keep it to 1-3 sentences. "
            "If you provide a list of suggestions or examples, ALWAYS use the '-' character for bullet points and NEVER use dashes (-)."
        )

        llm = get_llm(temperature=0.9)
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Admin Question: {masked_question}")
        ])

        return masker.unmask_text(response.content.strip(), mapping)
    except Exception as e:
        logger.error(f"Dynamic error generation failed: {e}")
        return static_message
