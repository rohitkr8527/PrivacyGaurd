from collections import Counter

import modal
import streamlit as st


st.set_page_config(
    page_title="PrivacyGuard",
    layout="wide",
    initial_sidebar_state="expanded",
)

SAMPLE_TEXT = """Dear Mr. Daniel Carter,

Please confirm that your email is daniel.carter@example.com and your phone number is +1-415-555-0198.
Your passport number is X1234567 and your date of birth is 14 March 1994.
Your current address is 27 Market Street, San Francisco, 94105.

Regards,
Support Team
"""

if "input_text" not in st.session_state:
    st.session_state.input_text = SAMPLE_TEXT


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("PrivacyGuard")
st.write(
    "Detect personally identifiable information and produce "
    "a privacy-safe redacted version of the text."
)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:
    st.header("About")

    st.write(
        "PrivacyGuard uses a QLoRA fine-tuned Qwen3-4B model "
        "for structured PII extraction and deterministic redaction."
    )

    st.divider()

    st.subheader("Model")
    st.write("Base model: Qwen3-4B")
    st.write("Fine-tuning: QLoRA")
    st.write("Supported PII types: 19")
    st.write("Inference: Modal L4")

    st.divider()

    st.subheader("Evaluation")
    st.metric(
        "OpenPII F1",
        "0.9535",
    )
    st.metric(
        "OpenPII Recall",
        "0.9508",
    )
    st.metric(
        "PII Leakage Rate",
        "0.0492",
    )

    st.caption(
        "Evaluation metrics are from the frozen 3,000-sample "
        "OpenPII test set."
    )


# ---------------------------------------------------------
# Input
# ---------------------------------------------------------

st.subheader("1. Enter text")

button_col_1, button_col_2, spacer = st.columns(
    [1, 1, 5]
)

with button_col_1:
    if st.button(
        "Load sample",
        use_container_width=True,
    ):
        st.session_state.input_text = SAMPLE_TEXT
        st.rerun()

with button_col_2:
    if st.button(
        "Clear",
        use_container_width=True,
    ):
        st.session_state.input_text = ""
        st.rerun()

text = st.text_area(
    "Text to scan",
    key="input_text",
    height=240,
    placeholder=(
        "Paste text containing names, email addresses, "
        "phone numbers, IDs, dates, addresses, or other PII."
    ),
    label_visibility="collapsed",
)

char_count = len(text)
st.caption(
    f"{char_count:,} / 12,000 characters"
)

run = st.button(
    "Detect and redact PII",
    type="primary",
    use_container_width=True,
)


# ---------------------------------------------------------
# Inference
# ---------------------------------------------------------

if run:
    if not text.strip():
        st.warning(
            "Enter some text before running detection."
        )
        st.stop()

    if len(text) > 12000:
        st.error(
            "The demo accepts up to 12,000 characters per request."
        )
        st.stop()

    try:
        backend = modal.Function.from_name(
            "privacyguard-demo",
            "detect_and_redact",
        )

        with st.spinner(
            "Analyzing text and generating redaction..."
        ):
            result = backend.remote(text)

    except Exception as exc:
        st.error(
            "The inference backend could not be reached."
        )

        with st.expander(
            "Technical details"
        ):
            st.code(str(exc))

        st.info(
            "Deploy the backend with:\n\n"
            "uv run modal deploy app/modal_inference.py"
        )
        st.stop()

    entities = result["entities"]

    st.divider()

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    st.subheader("2. Detection summary")

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)

    metric_1.metric(
        "PII entities",
        result["entity_count"],
    )

    metric_2.metric(
        "Unique PII types",
        len(
            {
                entity["type"]
                for entity in entities
            }
        ),
    )

    metric_3.metric(
        "Structured output",
        "Valid" if result["json_valid"] else "Invalid",
    )

    metric_4.metric(
        "Generation status",
        str(result["finish_reason"]).title(),
    )

    # -----------------------------------------------------
    # Text comparison
    # -----------------------------------------------------

    st.subheader("3. Privacy-safe output")

    original_col, redacted_col = st.columns(2)

    with original_col:
        with st.container(border=True):
            st.markdown("**Original text**")
            st.text_area(
                "Original text result",
                value=text,
                height=310,
                disabled=True,
                label_visibility="collapsed",
            )

    with redacted_col:
        with st.container(border=True):
            st.markdown("**Redacted text**")
            st.text_area(
                "Redacted text result",
                value=result["redacted_text"],
                height=310,
                disabled=True,
                label_visibility="collapsed",
            )

            st.download_button(
                "Download redacted text",
                data=result["redacted_text"],
                file_name="privacyguard_redacted.txt",
                mime="text/plain",
                use_container_width=True,
            )

    # -----------------------------------------------------
    # Entity inspection
    # -----------------------------------------------------

    st.subheader("4. Detected entities")

    if entities:
        display_rows = [
            {
                "Text": entity["text"],
                "Type": entity["type"],
                "Start": entity["start"],
                "End": entity["end"],
            }
            for entity in entities
        ]

        st.dataframe(
            display_rows,
            use_container_width=True,
            hide_index=True,
        )

        type_counts = Counter(
            entity["type"]
            for entity in entities
        )

        st.markdown("**Entity distribution**")

        distribution_rows = [
            {
                "PII Type": label,
                "Count": count,
            }
            for label, count
            in sorted(
                type_counts.items(),
                key=lambda item: (
                    -item[1],
                    item[0],
                ),
            )
        ]

        st.bar_chart(
            distribution_rows,
            x="PII Type",
            y="Count",
        )

    else:
        st.info(
            "No supported PII entities were detected."
        )

    # -----------------------------------------------------
    # Debug information
    # -----------------------------------------------------

    with st.expander(
        "Model output details"
    ):
        st.caption(
            "Raw structured response returned by the fine-tuned model."
        )
        st.code(
            result["raw_response"],
            language="json",
        )
