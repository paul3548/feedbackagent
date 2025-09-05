
import openai
import base64
from PIL import Image
import io
import streamlit as st
from typing import Optional, Dict, Any

class DFDAnalyzer:
    def __init__(self, api_key: str):
        """Initialize the DFD analyzer with OpenAI API key."""
        self.client = openai.OpenAI(api_key=api_key)
        
        # Customizable prompt components
        self.system_description = """
        You are analyzing a dataflow diagram (DFD) for a student learning exercise. 
        The system being modeled is a typical business information system with the following characteristics:
        - Processes that transform data (shown as circles or rounded rectangles)
        - Data stores that hold information (shown as open rectangles or parallel lines)
        - External entities that are sources or destinations of data (shown as squares)
        - Data flows that show movement of information (shown as labeled arrows)
        
        The student should demonstrate understanding of how data moves through a system 
        and transforms at each process step.
        """
        
        self.common_faults = """
        Check for these common DFD mistakes:
        
        1. **Direct datastore-to-datastore flows**: Data should NEVER flow directly between 
           two data stores without going through a process. This is the most critical error.
        
        2. **Unlabeled data flows**: All arrows should be labeled with what data is flowing.
        
        3. **Missing external entities**: Systems need to show where data originates and 
           where outputs go outside the system boundary.
        
        4. **Process naming**: Processes should use action verbs (e.g., "Validate Order", 
           "Calculate Total") not just nouns.
        
        5. **Unbalanced flows**: Processes should have both inputs and outputs. 
           A process can't just produce data from nothing.
        
        6. **Data store access**: Data stores should be accessed (read from or written to) 
           by processes, not directly by external entities.
        
        7. **Incorrect symbols**: Check that circles/ovals are used for processes, 
           rectangles for external entities, and parallel lines or open rectangles for data stores.
        """
        
        self.feedback_guidance = """
        Provide feedback in this structured format:
        
        **DIAGRAM IDENTIFICATION**
        - Confirm if this is a dataflow diagram
        - Rate confidence level (1-10)
        
        **CRITICAL ERRORS** (if any)
        - List any direct datastore-to-datastore connections
        - Identify missing essential elements
        
        **IMPROVEMENT AREAS**
        - Point out unlabeled flows
        - Suggest better process naming
        - Note missing external entities
        
        **STRENGTHS**
        - Highlight what the student did well
        - Acknowledge correct use of symbols and conventions
        
        **OVERALL ASSESSMENT**
        - Brief summary of diagram quality
        - Specific suggestions for improvement
        - Educational guidance for learning
        
        Keep feedback constructive and educational. Explain WHY each point matters 
        for creating effective dataflow diagrams.
        """
    
    def update_system_description(self, new_description: str):
        """Update the system description component."""
        self.system_description = new_description
    
    def update_common_faults(self, new_faults: str):
        """Update the common faults to check for."""
        self.common_faults = new_faults
    
    def update_feedback_guidance(self, new_guidance: str):
        """Update the feedback guidance format."""
        self.feedback_guidance = new_guidance
    
    def compose_prompt(self) -> str:
        """Compose the complete prompt from the three components."""
        prompt = f"""
{self.system_description}

{self.common_faults}

{self.feedback_guidance}

Please analyze the uploaded dataflow diagram image and provide detailed, educational feedback following the guidelines above.
"""
        return prompt.strip()
    
    def encode_image(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string for OpenAI API."""
        # Convert to RGB if necessary (handles RGBA, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save to bytes
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='JPEG', quality=95)
        img_buffer.seek(0)
        
        # Encode to base64
        img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
        return img_base64
    
    async def analyze_dfd(self, image: Image.Image, model: str = "gpt-4o") -> Dict[str, Any]:
        """
        Analyze a DFD image using OpenAI's vision model.
        
        Args:
            image: PIL Image object
            model: OpenAI model to use (gpt-4o, gpt-4o-mini, etc.)
        
        Returns:
            Dictionary with analysis results and metadata
        """
        try:
            # Encode image
            base64_image = self.encode_image(image)
            
            # Compose prompt
            prompt = self.compose_prompt()
            
            # Make API call
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"  # Use "high" for detailed analysis
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1500,
                temperature=0.1  # Low temperature for consistent analysis
            )
            
            # Extract response
            feedback = response.choices[0].message.content
            
            return {
                "success": True,
                "feedback": feedback,
                "model_used": model,
                "tokens_used": response.usage.total_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "feedback": "Analysis failed. Please try again."
            }
    
    def analyze_dfd_sync(self, image: Image.Image, model: str = "gpt-4o") -> Dict[str, Any]:
        """
        Synchronous version of analyze_dfd for use in Streamlit.
        """
        try:
            # Encode image
            base64_image = self.encode_image(image)
            
            # Compose prompt
            prompt = self.compose_prompt()
            
            # Make API call
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1500,
                temperature=0.1
            )
            
            # Extract response
            feedback = response.choices[0].message.content
            
            return {
                "success": True,
                "feedback": feedback,
                "model_used": model,
                "tokens_used": response.usage.total_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "feedback": f"Analysis failed: {str(e)}"
            }


# Streamlit integration example
def create_streamlit_interface():
    """Example Streamlit interface using the DFD analyzer."""
    
    st.title("📊 Dataflow Diagram Analyzer")
    st.write("Upload your DFD and get instant feedback!")
    
    # API key input (use st.secrets in production)
    if 'openai_api_key' not in st.session_state:
        api_key = st.text_input("Enter your OpenAI API Key:", type="password")
        if api_key:
            st.session_state.openai_api_key = api_key
    
    if 'openai_api_key' in st.session_state:
        # Initialize analyzer
        if 'analyzer' not in st.session_state:
            st.session_state.analyzer = DFDAnalyzer(st.session_state.openai_api_key)
        
        analyzer = st.session_state.analyzer
        
        # Optional: Allow prompt customization
        with st.expander("🔧 Customize Analysis Parameters"):
            st.subheader("System Description")
            new_system_desc = st.text_area(
                "Describe the type of system being modeled:",
                value=analyzer.system_description,
                height=100
            )
            
            st.subheader("Common Faults to Check")
            new_faults = st.text_area(
                "List common DFD mistakes to identify:",
                value=analyzer.common_faults,
                height=200
            )
            
            st.subheader("Feedback Format")
            new_guidance = st.text_area(
                "Specify how feedback should be structured:",
                value=analyzer.feedback_guidance,
                height=150
            )
            
            if st.button("Update Analysis Parameters"):
                analyzer.update_system_description(new_system_desc)
                analyzer.update_common_faults(new_faults)
                analyzer.update_feedback_guidance(new_guidance)
                st.success("Parameters updated!")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload your dataflow diagram:",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff']
        )
        
        if uploaded_file:
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Your DFD", use_column_width=True)
            
            # Model selection
            model = st.selectbox(
                "Choose OpenAI model:",
                ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
                index=0
            )
            
            # Analysis button
            if st.button("🔍 Analyze Diagram", type="primary"):
                with st.spinner("Analyzing your dataflow diagram..."):
                    result = analyzer.analyze_dfd_sync(image, model)
                
                if result["success"]:
                    st.success("✅ Analysis Complete!")
                    
                    # Display feedback
                    st.markdown("## 📋 Feedback")
                    st.markdown(result["feedback"])
                    
                    # Show usage stats
                    with st.expander("📊 Analysis Details"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Model Used", result["model_used"])
                        with col2:
                            st.metric("Total Tokens", result["tokens_used"])
                        with col3:
                            st.metric("Cost Estimate", f"${result['tokens_used'] * 0.00003:.4f}")
                else:
                    st.error(f"❌ Analysis failed: {result['error']}")
    else:
        st.warning("Please enter your OpenAI API key to begin analysis.")


if __name__ == "__main__":
    create_streamlit_interface()