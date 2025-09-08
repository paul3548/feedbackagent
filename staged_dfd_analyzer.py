import openai
import base64
from PIL import Image
import io
import streamlit as st
from typing import Optional, Dict, Any, Tuple
import hashlib
import json


class SelectiveCacheDFDAnalyzer:
    def __init__(self, api_key: str):
        """Initialize the DFD analyzer with selective caching."""
        self.client = openai.OpenAI(api_key=api_key)
        
        # Stage 1: Notation check and description generation
        self.stage1_prompt = """
        You are analyzing a dataflow diagram (DFD) to check notation and create a detailed description.
        
        TASK: Examine this DFD image and provide a structured analysis of its components.
        
        Please respond in this EXACT format:        
       
        **NOTATION CHECK**
        - Are correct symbols used for processes (circles/ovals)?
        - Are correct symbols used for external entities (squares/rectangles)?
        - Are correct symbols used for data stores (open rectangles/parallel lines)?
        
        **EXTERNAL ENTITIES**
        List all external entities found:
        - Entity 1: [Name]
        - Entity 2: [Name]
        (Continue for all entities found)
        
        **PROCESSES**
        List all processes found:
        - Process 1: [Name/Description]
        - Process 2: [Name/Description]
        (Continue for all processes found)
        
        **DATA STORES**
        List all data stores found:
        - Data Store 1: [Name]
        - Data Store 2: [Name]
        (Continue for all data stores found)
        
        **DATA FLOWS**
        List all data flows found:
        - Flow 1: From [Source] to [Destination], Label: [Label or "Unlabeled"]
        - Flow 2: From [Source] to [Destination], Label: [Label or "Unlabeled"]
        (Continue for all flows found)
        
         """
        
        # Stage 2: Error checking template
        self.stage2_prompt_template = """
        Based on the description of the DFD components provided, check for DFD errors and violations.
        
        COMPONENT DESCRIPTION:
        {description}

        Check for these specific errors:
        
        **CRITICAL ERRORS**
        1. Direct datastore-to-datastore flows (data flowing between stores without a process)
        2. Processes with no inputs or no outputs
        3. External entities connected directly to data stores
        4. Elements that do not belong in a DFD (e.g., control flows, UI elements)
        
        **LABELING ISSUES**
        1. Unlabeled data flows
        2. Labels on data flows that identify actions, processes or conditions
        3. Unlabeled processes or data stores
        4. Incorrect use of notation
               
        Do not check the diagram against the system description or comment on its accuracy or completeness.
        Report and explain significant errors found. Be succinct.
        """
        
        # Stage 3: System modeling assessment template
        self.stage3_prompt_template = """

       Evaluate how well the dfd description models the system described in the text
        
        COMPONENT DESCRIPTION:
        {description}   
     
        SYSTEM REQUIREMENTS:
        {system_description}
        
        Assess the following:
        
        **SYSTEM COVERAGE**
        - Does the diagram capture the processes?
        - Are all necessary data flows represented?
        - Are all stakeholders (external entities) identified?
                
        **LOGIC**
        - Is the sequence of operations logical?
        - Are the data flow directions correct?
         
        **OVERALL ASSESSMENT**
        - Provide specific suggestions for improvement
        - Highlight what the student did well
        
        Keep feedback constructive and succinct
        """
        
        # Default system description
        self.system_description = """
        A typical business information system that should demonstrate proper data flow 
        between external entities, processes, and data stores representing realistic 
        business operations.
        """
        
        # SELECTIVE CACHE: Only cache prompt templates, not content
        self.prompt_cache = {}
        self._cache_prompts()
    
    def _cache_prompts(self):
        """Cache the static prompt templates (not content-specific)."""
        self.prompt_cache['stage1'] = self.stage1_prompt
        self.prompt_cache['stage2_template'] = self.stage2_prompt_template
        self.prompt_cache['stage3_template'] = self.stage3_prompt_template
     #   st.info("📝 Prompt templates cached (reusable across all students)")
    
    def update_system_description(self, new_description: str):
        """Update the system description."""
        self.system_description = new_description
        # Re-cache prompts if system description is used in templates
        self._cache_prompts()
    
    def _encode_image(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string for OpenAI API."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='JPEG', quality=95)
        img_buffer.seek(0)
        
        img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
        return img_base64
    
    def _call_openai(self, prompt: str, image_base64: str = None, model: str = "gpt-4o") -> Dict[str, Any]:
        """Make a call to OpenAI API with error handling."""
        try:
            if image_base64:
                # Call with image
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }]
            else:
                # Text-only call
                messages = [{"role": "user", "content": prompt}]
            
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=8000
                
            )
            
            return {
                "success": True,
                "content": response.choices[0].message.content,
                "tokens_used": response.usage.total_tokens,
                "model": model
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "content": f"Analysis failed: {str(e)}"
            }
    
    def stage1_analyze_notation(self, image: Image.Image, model: str = "gpt-4o") -> Dict[str, Any]:
        """Stage 1: Check notation and generate component description"""
        
        # Use cached prompt template
        prompt = self.prompt_cache['stage1']
        
        # Always make fresh API call for each image
        image_base64 = self._encode_image(image)
        #st.info("🔍 Analyzing image notation (fresh analysis for this student)")
        
        result = self._call_openai(prompt, image_base64, model)
        return result
    
    def stage2_check_errors(self, description: str, model: str = "gpt-4o") -> Dict[str, Any]:
        """Stage 2: Check for DFD errors """
        
        # Use cached template and format with specific content
        prompt_template = self.prompt_cache['stage2_template']
        prompt = prompt_template.format(
            description=description,
            system_description=self.system_description
        )
        
        # Always make fresh API call for each student's description
        st.info("⚠️ Checking errors")
        
        result = self._call_openai(prompt, None, model)
        return result
    
    def stage3_assess_system(self, description: str, error_analysis: str, model: str = "gpt-4o") -> Dict[str, Any]:
        """Stage 3: Assess system modeling"""
        
        # Use cached template and format with specific content
        prompt_template = self.prompt_cache['stage3_template']
        prompt = prompt_template.format(
            description=description,
            error_analysis=error_analysis,
            system_description=self.system_description
        )
        
        # Always make fresh API call for each student's analysis
        #st.info("🎯 Assessing system modeling (fresh analysis for this student)")
        
        result = self._call_openai(prompt, None, model)
        return result
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about what's cached."""
        return {
            "cached_prompts": list(self.prompt_cache.keys()),
            "prompt_count": len(self.prompt_cache),
            "system_description_length": len(self.system_description)
        }
    
    def clear_prompt_cache(self):
        """Clear the prompt cache (rarely needed)."""
        self.prompt_cache.clear()
        st.warning("Prompt cache cleared - will need to re-initialize")
    
    def analyze_complete(self, image: Image.Image, model: str = "gpt-4o") -> Dict[str, Any]:
        """Run all three stages of analysis with selective caching."""
        
        total_tokens = 0
        results = {}
        
        # Stage 1: Fresh analysis for each image
        st.info("Starting Stage 1...")
        stage1_result = self.stage1_analyze_notation(image, model)
        if not stage1_result["success"]:
            return {
                "success": False,
                "error": f"Stage 1 failed: {stage1_result['error']}",
                "stage": 1
            }
        
        results["stage1"] = stage1_result["content"]
        total_tokens += stage1_result["tokens_used"]
        
        # Stage 2: Fresh analysis for each description
        st.info("Starting Stage 2...")
        stage2_result = self.stage2_check_errors(stage1_result["content"], model)
        if not stage2_result["success"]:
            return {
                "success": False,
                "error": f"Stage 2 failed: {stage2_result['error']}",
                "stage": 2
            }
        
        results["stage2"] = stage2_result["content"]
        total_tokens += stage2_result["tokens_used"]
        
        # Stage 3: Fresh analysis for each assessment
        st.info("Starting Stage 3...")
        stage3_result = self.stage3_assess_system(
            stage1_result["content"], 
            stage2_result["content"], 
            model
        )
        if not stage3_result["success"]:
            return {
                "success": False,
                "error": f"Stage 3 failed: {stage3_result['error']}",
                "stage": 3
            }
        
        results["stage3"] = stage3_result["content"]
        total_tokens += stage3_result["tokens_used"]
        
        return {
            "success": True,
            "results": results,
            "total_tokens": total_tokens,
            "model_used": model
        }

class StagedDFDAnalyzer:
    def __init__(self, api_key: str):
        """Initialize the staged DFD analyzer with OpenAI API key."""
        self.client = openai.OpenAI(api_key=api_key)
        
        # Stage 1: Notation check and description generation
        self.stage1_prompt = """
        You are analyzing a dataflow diagram (DFD) to check notation and create a detailed description.
        
        TASK: Examine this DFD image and provide a structured analysis of its components.
        
        Please respond in this EXACT format:       
       
        **NOTATION CHECK**
        - Are correct symbols used for processes (circles/ovals)?
        - Are correct symbols used for external entities (squares/rectangles)?
        - Are correct symbols used for data stores (open rectangles/parallel lines)?
          
        **EXTERNAL ENTITIES**
        List all external entities found:
        - Entity 1: [Name]
        - Entity 2: [Name]
        (Continue for all entities found)
        
        **PROCESSES**
        List all processes found:
        - Process 1: [Name/Description]
        - Process 2: [Name/Description]
        (Continue for all processes found)
        
        **DATA STORES**
        List all data stores found:
        - Data Store 1: [Name]
        - Data Store 2: [Name]
        (Continue for all data stores found)
        
        **DATA FLOWS**
        List all data flows found:
        - Flow 1: From [Source] to [Destination], Label: [Label or "Unlabeled"]
        - Flow 2: From [Source] to [Destination], Label: [Label or "Unlabeled"]
        (Continue for all flows found)

        """
        
        # Stage 2: Error checking
        self.stage2_prompt = """
        Based on the detailed description of the DFD components provided, check for common DFD errors and violations.
        
        COMPONENT DESCRIPTION:
        {description}
      
        Check for these specific errors:
        
        **CRITICAL ERRORS**
        1. Direct datastore-to-datastore flows (data flowing between stores without a process)
        2. Processes with no inputs or no outputs
        3. External entities connected directly to data stores
        4. Elements that do not belong in a DFD (e.g., control flows, UI elements)
e        
        **LABELING ISSUES**
        1. Unlabeled data flows
        2. Labels on data flows that identify actions, processes or conditions
        3. Unlabeled processes or data stores
        4. Incorrect use of notation
                     
        Do not check the diagram against the system description or comment on its accuracy or completeness.
        Report and explain significant errors found. Be succinct.
        """
        
        # Stage 3: System modeling assessment
        self.stage3_prompt = """
        Evaluate how well the dfd description models the system described in the text
        
        COMPONENT DESCRIPTION:
        {description}   
     
        SYSTEM REQUIREMENTS:
        {system_description}
        
        Assess the following:
        
        **SYSTEM COVERAGE**
        - Does the diagram capture the processes?
        - Are all necessary data flows represented?
        - Are all stakeholders (external entities) identified?
                
        **LOGIC**
        - Is the sequence of operations logical?
        - Are the data flow directions correct?
         
        **OVERALL ASSESSMENT**
        - Provide specific suggestions for improvement
        - Highlight what the student did well
        
        Keep feedback constructive and succinct
        """
        
        # Default system description
        self.system_description = """
        A typical business information system that should demonstrate proper data flow 
        between external entities, processes, and data stores representing realistic 
        business operations.
        """
        
        # Cache for avoiding duplicate API calls
        self.image_cache = {}
    
    def update_system_description(self, new_description: str):
        """Update the system description for stages 2 and 3."""
        self.system_description = new_description
    
    def _get_image_hash(self, image: Image.Image) -> str:
        """Generate a hash of the image for caching."""
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return hashlib.md5(img_bytes.read()).hexdigest()
    
    def _encode_image(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string for OpenAI API."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='JPEG', quality=95)
        img_buffer.seek(0)
        
        img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
        return img_base64
    
    def _call_openai(self, prompt: str, image_base64: str = None, model: str = "gpt-4o") -> Dict[str, Any]:
        """Make a call to OpenAI API with error handling."""
        try:
            if image_base64:
                # Call with image
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }]
            else:
                # Text-only call
                messages = [{"role": "user", "content": prompt}]
            
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=8000
                
            )
            
            return {
                "success": True,
                "content": response.choices[0].message.content,
                "tokens_used": response.usage.total_tokens,
                "model": model
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "content": f"Analysis failed: {str(e)}"
            }
    
    def stage1_analyze_notation(self, image: Image.Image, model: str = "gpt-4o") -> Dict[str, Any]:
        """Stage 1: Check notation and generate component description."""
        
        # Check cache first
        image_hash = self._get_image_hash(image)
        cache_key = f"stage1_{image_hash}_{model}"
        
        if cache_key in self.image_cache:
            st.info("🔄 Using cached notation analysis")
            return self.image_cache[cache_key]
        
        # Encode image
        image_base64 = self._encode_image(image)
        
        # Make API call
        result = self._call_openai(self.stage1_prompt, image_base64, model)
        
        # Cache the result
        if result["success"]:
            self.image_cache[cache_key] = result
        
        return result
    
    def stage2_check_errors(self, description: str, model: str = "gpt-4o") -> Dict[str, Any]:
        """Stage 2: Check for DFD errors based on the description."""
        
        # Create cache key based on description and system description
        cache_key = f"stage2_{hashlib.md5((description + self.system_description).encode()).hexdigest()}_{model}"
        
        if cache_key in self.image_cache:
            st.info("🔄 Using cached error analysis")
            return self.image_cache[cache_key]
        
        # Format prompt with description
        prompt = self.stage2_prompt.format(
            description=description,
            system_description=self.system_description
        )
        
        # Make API call (text only)
        result = self._call_openai(prompt, None, model)
        
        # Cache the result
        if result["success"]:
            self.image_cache[cache_key] = result
        
        return result
    
    def stage3_assess_system(self, description: str, error_analysis: str, model: str = "gpt-4o") -> Dict[str, Any]:
        """Stage 3: Assess how well the diagram models the system."""
        
        # Create cache key
        cache_key = f"stage3_{hashlib.md5((description + error_analysis + self.system_description).encode()).hexdigest()}_{model}"
        
        if cache_key in self.image_cache:
            st.info("🔄 Using cached system assessment")
            return self.image_cache[cache_key]
        
        # Format prompt
        prompt = self.stage3_prompt.format(
            description=description,
            error_analysis=error_analysis,
            system_description=self.system_description
        )
        
        # Make API call (text only)
        result = self._call_openai(prompt, None, model)
        
        # Cache the result
        if result["success"]:
            self.image_cache[cache_key] = result
        
        return result
    
    def analyze_complete(self, image: Image.Image, model: str = "gpt-4o") -> Dict[str, Any]:
        """Run all three stages of analysis."""
        
        total_tokens = 0
        results = {}
        
        # Stage 1: Notation and Description
        stage1_result = self.stage1_analyze_notation(image, model)
        if not stage1_result["success"]:
            return {
                "success": False,
                "error": f"Stage 1 failed: {stage1_result['error']}",
                "stage": 1
            }
        
        results["stage1"] = stage1_result["content"]
        total_tokens += stage1_result["tokens_used"]
        
        # Stage 2: Error Checking
        stage2_result = self.stage2_check_errors(stage1_result["content"], model)
        if not stage2_result["success"]:
            return {
                "success": False,
                "error": f"Stage 2 failed: {stage2_result['error']}",
                "stage": 2
            }
        
        results["stage2"] = stage2_result["content"]
        total_tokens += stage2_result["tokens_used"]
        
        # Stage 3: System Assessment
        stage3_result = self.stage3_assess_system(
            stage1_result["content"], 
            stage2_result["content"], 
            model
        )
        if not stage3_result["success"]:
            return {
                "success": False,
                "error": f"Stage 3 failed: {stage3_result['error']}",
                "stage": 3
            }
        
        results["stage3"] = stage3_result["content"]
        total_tokens += stage3_result["tokens_used"]
        
        return {
            "success": True,
            "results": results,
            "total_tokens": total_tokens,
            "model_used": model
        }
    
    def clear_cache(self):
        """Clear the analysis cache."""
        self.image_cache.clear()


# Streamlit app interface
def create_staged_interface():
    """Create Streamlit interface for staged analysis."""
    
    st.title("📊 Staged DFD Analyzer")
    st.write("Three-stage analysis: Notation → Errors → System Assessment")
    
    # Initialize analyzer
    if 'staged_analyzer' not in st.session_state:
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
            st.session_state.staged_analyzer = StagedDFDAnalyzer(api_key)
            
            # Set default system description
            default_scenario = """
            The NHS electronic prescribing system handles repeat prescriptions from review to dispensing.
            The system should show processes for prescription review, regimen creation, prescription decisions,
            and medicine dispensing, with appropriate data flows between patients, prescribers, NHS Spine,
            and pharmacy systems.
            """
            st.session_state.staged_analyzer.update_system_description(default_scenario)
            
        except Exception as e:
            st.error(f"Failed to initialize analyzer: {e}")
            return
    
    analyzer = st.session_state.staged_analyzer
    
    # System description
    with st.expander("📋 System Description"):
        current_desc = analyzer.system_description
        new_desc = st.text_area("System being modeled:", value=current_desc, height=100)
        if st.button("Update System Description"):
            analyzer.update_system_description(new_desc)
            st.success("System description updated!")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload DFD", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Your DFD", use_column_width=True)
        
        # Model selection
        model_choice = st.selectbox(
            "Choose model:",
            ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
            index=0
        )
        
        # Analysis options
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 Run Complete Analysis", type="primary"):
                with st.spinner("Running three-stage analysis..."):
                    result = analyzer.analyze_complete(image, model_choice)
                
                if result["success"]:
                    st.success("✅ Complete Analysis Finished!")
                    
                    # Display all results
                    st.markdown("## 📋 Stage 1: Notation & Description")
                    st.markdown(result["results"]["stage1"])
                    
                    st.markdown("## ⚠️ Stage 2: Error Analysis")
                    st.markdown(result["results"]["stage2"])
                    
                    st.markdown("## 🎯 Stage 3: System Assessment")
                    st.markdown(result["results"]["stage3"])
                    
                    # Usage stats
                    with st.expander("📊 Usage Statistics"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Model", result["model_used"])
                        with col2:
                            st.metric("Total Tokens", result["total_tokens"])
                        with col3:
                            cost = result["total_tokens"] * 0.00003
                            st.metric("Est. Cost", f"${cost:.4f}")
                else:
                    st.error(f"❌ Analysis failed at stage {result.get('stage', '?')}: {result['error']}")
        
        with col2:
            if st.button("🔄 Clear Cache"):
                analyzer.clear_cache()
                st.success("Cache cleared!")
        
        # Individual stage controls
        st.markdown("### Individual Stage Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Stage 1: Notation"):
                with st.spinner("Analyzing notation..."):
                    result = analyzer.stage1_analyze_notation(image, model_choice)
                if result["success"]:
                    st.markdown("#### Notation Analysis")
                    st.markdown(result["content"])
                else:
                    st.error(f"Stage 1 failed: {result['error']}")
        
        with col2:
            if st.button("Stage 2: Errors") and 'stage1_result' in st.session_state:
                with st.spinner("Checking errors..."):
                    result = analyzer.stage2_check_errors(st.session_state.stage1_result, model_choice)
                if result["success"]:
                    st.markdown("#### Error Analysis")
                    st.markdown(result["content"])
                else:
                    st.error(f"Stage 2 failed: {result['error']}")
        
        with col3:
            if st.button("Stage 3: System") and 'stage1_result' in st.session_state and 'stage2_result' in st.session_state:
                with st.spinner("Assessing system..."):
                    result = analyzer.stage3_assess_system(
                        st.session_state.stage1_result,
                        st.session_state.stage2_result,
                        model_choice
                    )
                if result["success"]:
                    st.markdown("#### System Assessment")
                    st.markdown(result["content"])
                else:
                    st.error(f"Stage 3 failed: {result['error']}")


if __name__ == "__main__":
    create_staged_interface()