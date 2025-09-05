import streamlit as st
from dfd_analyzer import DFDAnalyzer  # Import the class
from PIL import Image

st.title("📊 Dataflow Diagram Analyzer")

# Initialize analyzer
if 'analyzer' not in st.session_state:
    api_key = st.secrets["OPENAI_API_KEY"]  # Store in Streamlit secrets
    st.session_state.analyzer = DFDAnalyzer(api_key)

# File uploader
uploaded_file = st.file_uploader("Upload DFD", type=['png', 'jpg', 'jpeg'])

# Check if file was uploaded (no session state needed)
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Your DFD")
    
    if st.button("Analyze Diagram"):
        with st.spinner("Analyzing..."):
            result = st.session_state.analyzer.analyze_dfd_sync(image)
        
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