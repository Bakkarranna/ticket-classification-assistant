import os
import requests
import streamlit as st

st.set_page_config(page_title="Campus IT Ticket Assistant", page_icon="🎫", layout="centered")
st.title("Campus IT Ticket Classification Assistant")
st.caption("University IT help desk • Local demo mode • Human review required")

api_url = os.getenv("API_URL", "http://127.0.0.1:8000")
with st.sidebar:
    st.subheader("Settings")
    st.write("Backend:", api_url)
    prompt_version = st.selectbox("Prompt version", ["v2", "v1"], index=0)
    st.info("This project uses a deterministic local mock, not a live LLM. Results are suggestions only.")

ticket = st.text_area("Describe your IT issue", height=160, placeholder="Example: I cannot connect to campus Wi-Fi...")
col1, col2 = st.columns([1, 1])
with col1:
    analyze = st.button("Analyze ticket", type="primary", use_container_width=True)
with col2:
    st.button("Clear", use_container_width=True, on_click=lambda: st.session_state.update({"ticket_input": ""}))

if analyze:
    if not ticket.strip():
        st.error("Please enter a ticket before analyzing.")
    elif len(ticket.strip()) < 5:
        st.error("Ticket must contain at least 5 characters.")
    else:
        try:
            response = requests.post(f"{api_url}/analyze", json={
                "ticket": ticket, "prompt_version": prompt_version, "mode": "local_mock"
            }, timeout=10)
            if response.status_code == 200:
                result = response.json()
                st.success("Analysis complete. Please review before routing.")
                st.metric("Suggested category", result["category"].replace("_", " "))
                st.write("**Summary**")
                st.write(result["summary"])
                st.caption(f"Local similarity score: {result['confidence']:.3f} • Mode: {result['mode']}")
                st.warning("Human review required. No account, payment, or device actions are performed.")
            else:
                try:
                    detail = response.json().get("detail", response.text)
                except Exception:
                    detail = response.text
                st.error(f"Backend returned {response.status_code}: {detail}")
        except requests.exceptions.ConnectionError:
            st.error("Backend unavailable. Start FastAPI with: uvicorn app.api:app --reload")
        except requests.exceptions.Timeout:
            st.error("The backend request timed out. Please try again.")
        except Exception as exc:
            st.error(f"Unexpected frontend error: {exc}")

with st.expander("Try a controlled failure"):
    st.write("Stop the FastAPI server, then click Analyze. The frontend should show a backend-unavailable message.")
