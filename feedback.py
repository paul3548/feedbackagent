import streamlit as st
from dfd_analyzer import DFDAnalyzer  # Import the class
from PIL import Image

default_scenario = "The NHS electronic prescribing system handles the repeat prescriptions from review to dispensing. If, at a consultation, a patient and their GP agree that the patient should receive a repeat prescription with a prescribed drug, the GP will create a regimen for the prescription to be uploaded to the NHS Spine.This is a secure system and the GP will have to enter a PIN to access the system. The patient must nominate a pharmacy of their choice to receive the prescription.         When the patient goes to the pharmacy to collect their medication, the pharmacist downloads the prescription from the Spine. The pharmacist must ask the patient four questions (have you seen a health professional since the last repeat, have you started any new treatment, have you had any problems, is there an item on your prescription you no longer need). The pharmacist can also contact the GP if they have any concerns or questions. If the checks are satisfied, the treatment is dispensed, the patient is given any additional advice that may be required and record stored on both the pharmacy system and the NHS Spine."
model = "gpt-4o-mini"

st.title("📊 Dataflow Diagram Analyzer")

st.write("Upload a dataflow diagram (DFD) image for the following scenario to receive feedback on its structure and common pitfalls.")

st.write( default_scenario)

# Initialize analyzer
if 'analyzer' not in st.session_state:
    api_key = st.secrets["OPENAI_API_KEY"]  # Store in Streamlit secrets
    st.session_state.analyzer = DFDAnalyzer(api_key)

    st.session_state.analyzer.update_system_description(default_scenario)

# File uploader
uploaded_file = st.file_uploader("Upload DFD", type=['png', 'jpg', 'jpeg'])

# Check if file was uploaded (no session state needed)
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Your DFD")
    
    if st.button("Analyze Diagram"):
        with st.spinner("Analyzing..."):
            result = st.session_state.analyzer.analyze_dfd_sync(image, model)
        
        if result["success"]:
            st.success("✅ Analysis Complete!")
            st.markdown("## 📋 Analysis Results")
            st.markdown(result["feedback"])
            
            # Optional: Show usage stats
            with st.expander("📊 Usage Details"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Model", result["model_used"])
                with col2:
                    st.metric("Tokens", result["tokens_used"])
                with col3:
                    cost = result["tokens_used"] * 0.00003
                    st.metric("Est. Cost", f"${cost:.4f}")
        else:
            st.error(f"❌ Analysis failed: {result['error']}")
else:
    st.info("👆 Please upload a dataflow diagram to begin analysis")
