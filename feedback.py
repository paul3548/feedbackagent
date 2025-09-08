import streamlit as st
from PIL import Image
from staged_dfd_analyzer import SelectiveCacheDFDAnalyzer

# CONFIGURATION - Set your model choice here
MODEL_CHOICE = "gpt-5"  # Change this to: "gpt-4o", "gpt-4o-mini", or "gpt-4-turbo"

# System scenario
default_scenario = """The NHS electronic prescribing system handles the repeat prescriptions from review to dispensing. If, at a consultation, a patient and their GP agree that the patient should receive a repeat prescription with a prescribed drug, the GP will create a regimen for the prescription to be uploaded to the NHS Spine. This is a secure system and the GP will have to enter a PIN to access the system. The patient must nominate a pharmacy of their choice to receive the prescription. When the patient goes to the pharmacy to collect their medication, the pharmacist downloads the prescription from the Spine. The pharmacist must ask the patient four questions (have you seen a health professional since the last repeat, have you started any new treatment, have you had any problems, is there an item on your prescription you no longer need). The pharmacist can also contact the GP if they have any concerns or questions. If the checks are satisfied, the treatment is dispensed, the patient is given any additional advice that may be required and record stored on both the pharmacy system and the NHS Spine."""

st.title("📊 DFD Analyzer - Stage by Stage")
st.write(f"**Model:** {MODEL_CHOICE}")
st.write("Upload a DFD and run each analysis stage individually.")

# Display scenario
with st.expander("📋 System Scenario"):
    st.write(default_scenario)

# Initialize analyzer
if 'analyzer' not in st.session_state:
    try:
        # Try secrets first, fallback to user input
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except:
            st.warning("🔑 Enter your OpenAI API Key:")
            api_key = st.text_input("API Key:", type="password")
            if not api_key:
                st.stop()
        
        st.session_state.analyzer = SelectiveCacheDFDAnalyzer(api_key)
        st.session_state.analyzer.update_system_description(default_scenario)
        st.success("✅ Analyzer ready!")
    except Exception as e:
        st.error(f"Initialization failed: {e}")
        st.stop()

# File uploader
uploaded_file = st.file_uploader("Upload DFD Image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Your DFD", width=400)
    
    analyzer = st.session_state.analyzer
    
    st.markdown("---")
    
    # Stage 1: Notation and Description
    st.markdown("### Stage 1: Check Notation & Generate Description")
    if st.button("🔍 Run Stage 1", key="stage1"):
        with st.spinner("Analyzing notation and listing components..."):
            result = analyzer.stage1_analyze_notation(image, MODEL_CHOICE)
        
        if result["success"]:
            st.success("✅ Stage 1 Complete!")
            st.markdown(result["content"])
            st.session_state.stage1_result = result["content"]
            
            # Show usage
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Tokens", result["tokens_used"])
            with col2:
                cost = result["tokens_used"] * 0.00003
                st.metric("Cost", f"${cost:.4f}")
        else:
            st.error(f"Stage 1 failed: {result['error']}")
    
    # Stage 2: Error Analysis

# Stage 2: Error Analysis
st.markdown("### Stage 2: Check for Errors")

if st.button("⚠️ Run Stage 2", key="stage2", disabled='stage1_result' not in st.session_state):
    if 'stage1_result' in st.session_state:
        with st.spinner("Checking for DFD errors..."):
            result = analyzer.stage2_check_errors(st.session_state.stage1_result, MODEL_CHOICE)
        
        if result["success"]:
            st.success("✅ Stage 2 Complete!")
            st.markdown(result["content"])
            st.session_state.stage2_result = result["content"]
            
            # Show usage
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Tokens", result["tokens_used"])
            with col2:
                cost = result["tokens_used"] * 0.00003
                st.metric("Cost", f"${cost:.4f}")
        else:
            st.error(f"Stage 2 failed: {result['error']}")
    else:
        st.warning("Run Stage 1 first!")

if 'stage1_result' not in st.session_state:
    st.info("👆 Complete Stage 1 to unlock Stage 2")
    
# Stage 3: System Assessment
st.markdown("### Stage 3: Assess System Modeling")

if st.button("🎯 Run Stage 3", key="stage3", disabled='stage2_result' not in st.session_state):
    if 'stage1_result' in st.session_state and 'stage2_result' in st.session_state:
        with st.spinner("Assessing system modeling quality..."):
            result = analyzer.stage3_assess_system(
                st.session_state.stage1_result,
                st.session_state.stage2_result,
                MODEL_CHOICE
            )
        
        if result["success"]:
            st.success("✅ Stage 3 Complete!")
            st.markdown(result["content"])
            
            # Show usage
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Tokens", result["tokens_used"])
            with col2:
                cost = result["tokens_used"] * 0.00003
                st.metric("Cost", f"${cost:.4f}")
        else:
            st.error(f"Stage 3 failed: {result['error']}")
    else:
        st.warning("Complete Stages 1 and 2 first!")

if 'stage2_result' not in st.session_state:
    st.info("👆 Complete Stages 1 & 2 to unlock Stage 3")
    
    # Reset button
    st.markdown("---")
    if st.button("🔄 Reset All Stages"):
        for key in ['stage1_result', 'stage2_result']:
            if key in st.session_state:
                del st.session_state[key]
        st.success("All stages reset!")
        st.rerun()

else:
    st.info("👆 Upload a DFD image to begin analysis")
    
    # Show stage information
    with st.expander("ℹ️ About the Stages"):
        st.markdown("""
        **Stage 1: Notation & Description**
        - Checks DFD notation correctness
        - Lists all entities, processes, data stores
        - Documents data flows and labels
        
        **Stage 2: Error Analysis**
        - Identifies common DFD mistakes
        - Checks for critical errors
        - Analyzes structural problems
        
        **Stage 3: System Assessment** 
        - Evaluates business logic
        - Assesses completeness
        - Provides improvement suggestions
        """)

# Show current model in sidebar
st.sidebar.markdown(f"**Current Model:** `{MODEL_CHOICE}`")
st.sidebar.markdown("To change model, edit the `MODEL_CHOICE` variable in the code.")

if 'stage1_result' in st.session_state or 'stage2_result' in st.session_state:
    st.sidebar.markdown("**Progress:**")
    if 'stage1_result' in st.session_state:
        st.sidebar.markdown("✅ Stage 1 Complete")
    if 'stage2_result' in st.session_state:
        st.sidebar.markdown("✅ Stage 2 Complete")