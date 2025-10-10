import streamlit as st
from PIL import Image
from staged_dfd_analyzer import SelectiveCacheDFDAnalyzer
import re

# CONFIGURATION - Set your model choice here
MODEL_CHOICE = "gpt-5"  # Change this to: "gpt-4o", "gpt-4o-mini", or "gpt-4-turbo"

# System scenario
default_scenario = """The NHS electronic prescribing system handles the repeat prescriptions from review to dispensing. If, at a consultation, a patient and their GP agree that the patient should receive a repeat prescription with a prescribed drug, the GP will create a regimen for the prescription to be uploaded to the NHS Spine. This is a secure system and the GP will have to enter a PIN to access the system. The patient must nominate a pharmacy of their choice to receive the prescription. When the patient goes to the pharmacy to collect their medication, the pharmacist downloads the prescription from the Spine. The pharmacist must ask the patient four questions (have you seen a health professional since the last repeat, have you started any new treatment, have you had any problems, is there an item on your prescription you no longer need). The pharmacist can also contact the GP if they have any concerns or questions. If the checks are satisfied, the treatment is dispensed, the patient is given any additional advice that may be required and record stored on both the pharmacy system and the NHS Spine."""

def parse_stage1_result(content):
    """Parse Stage 1 result into structured components."""
    parsed = {
        'notation_check': '',
        'external_entities': [],
        'processes': [],
        'data_stores': [],
        'data_flows': []
    }
    
    lines = content.split('\n')
    current_section = None
    
    for line in lines:
        line = line.strip()
        
        if '**NOTATION CHECK**' in line:
            current_section = 'notation_check'
        elif '**EXTERNAL ENTITIES**' in line:
            current_section = 'external_entities'
        elif '**PROCESSES**' in line:
            current_section = 'processes'
        elif '**DATA STORES**' in line:
            current_section = 'data_stores'
        elif '**DATA FLOWS**' in line:
            current_section = 'data_flows'
        elif line.startswith('-') and current_section:
            item = line[1:].strip()
            if current_section == 'notation_check':
                parsed['notation_check'] += line + '\n'
            elif current_section in parsed and isinstance(parsed[current_section], list):
                parsed[current_section].append(item)
        elif current_section == 'notation_check' and line:
            parsed['notation_check'] += line + '\n'
    
    return parsed

def reconstruct_stage1_content(parsed_data):
    """Reconstruct Stage 1 content from edited components."""
    content = "**NOTATION CHECK**\n"
    content += parsed_data['notation_check'] + "\n\n"
    
    content += "**EXTERNAL ENTITIES**\n"
    content += "List all external entities found:\n"
    for i, entity in enumerate(parsed_data['external_entities'], 1):
        content += f"- Entity {i}: {entity}\n"
    
    content += "\n**PROCESSES**\n"
    content += "List all processes found:\n"
    for i, process in enumerate(parsed_data['processes'], 1):
        content += f"- Process {i}: {process}\n"
    
    content += "\n**DATA STORES**\n"
    content += "List all data stores found:\n"
    for i, store in enumerate(parsed_data['data_stores'], 1):
        content += f"- Data Store {i}: {store}\n"
    
    content += "\n**DATA FLOWS**\n"
    content += "List all data flows found:\n"
    for i, flow in enumerate(parsed_data['data_flows'], 1):
        content += f"- Flow {i}: {flow}\n"
    
    return content

st.title("📊 DFD Analyzer - Stage by Stage")
st.write(f"**Model:** {MODEL_CHOICE}")
st.write("Upload a DFD and run each analysis stage individually.")

# Display and edit scenario
with st.expander("📋 System Scenario", expanded=True):
    st.markdown("**Edit the system scenario below:**")
    st.info("💡 Paste your own scenario here to analyze different DFD problems")
    
    # Initialize scenario in session state if not present
    if 'current_scenario' not in st.session_state:
        st.session_state.current_scenario = default_scenario
    
    # Editable text area
    edited_scenario = st.text_area(
        "System Description:",
        value=st.session_state.current_scenario,
        height=200,
        key="scenario_input"
    )
    
    # Update button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("💾 Update Scenario"):
            st.session_state.current_scenario = edited_scenario
            st.session_state.analyzer.update_system_description(edited_scenario)
            st.success("✅ Scenario updated!")
            st.rerun()
    with col2:
        if st.button("🔄 Reset to Default"):
            st.session_state.current_scenario = default_scenario
            st.session_state.analyzer.update_system_description(default_scenario)
            st.success("✅ Reset to default scenario!")
            st.rerun()

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
            
            # Parse and store the result
            st.session_state.stage1_raw = result["content"]
            st.session_state.stage1_parsed = parse_stage1_result(result["content"])
            st.session_state.stage1_edited = False
            
            # Show usage
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Tokens", result["tokens_used"])
            with col2:
                cost = result["tokens_used"] * 0.00003
                st.metric("Cost", f"${cost:.4f}")
        else:
            st.error(f"Stage 1 failed: {result['error']}")
    
    # Edit Stage 1 Results
    if 'stage1_parsed' in st.session_state:
        st.markdown("---")
        st.markdown("### ✏️ Edit Stage 1 Results")
        st.info("Review and edit the components before running Stage 2. Add or remove items as needed.")
        
        parsed = st.session_state.stage1_parsed
        
        # Notation Check (read-only display)
        with st.expander("📝 Notation Check", expanded=False):
            st.text_area("Notation observations:", value=parsed['notation_check'], height=100, disabled=True, key="notation_display")
        
        # External Entities
        st.markdown("#### 🔷 External Entities")
        col1, col2 = st.columns([4, 1])
        with col1:
            new_entity = st.text_input("Add new entity:", key="new_entity")
        with col2:
            if st.button("➕ Add", key="add_entity"):
                if new_entity.strip():
                    parsed['external_entities'].append(new_entity.strip())
                    st.session_state.stage1_edited = True
                    st.rerun()
        
        entities_to_remove = []
        for i, entity in enumerate(parsed['external_entities']):
            col1, col2 = st.columns([4, 1])
            with col1:
                edited = st.text_input(f"Entity {i+1}:", value=entity, key=f"entity_{i}")
                if edited != entity:
                    parsed['external_entities'][i] = edited
                    st.session_state.stage1_edited = True
            with col2:
                if st.button("🗑️", key=f"del_entity_{i}"):
                    entities_to_remove.append(i)
                    st.session_state.stage1_edited = True
        
        for idx in sorted(entities_to_remove, reverse=True):
            parsed['external_entities'].pop(idx)
        if entities_to_remove:
            st.rerun()
        
        # Processes
        st.markdown("#### ⚙️ Processes")
        col1, col2 = st.columns([4, 1])
        with col1:
            new_process = st.text_input("Add new process:", key="new_process")
        with col2:
            if st.button("➕ Add", key="add_process"):
                if new_process.strip():
                    parsed['processes'].append(new_process.strip())
                    st.session_state.stage1_edited = True
                    st.rerun()
        
        processes_to_remove = []
        for i, process in enumerate(parsed['processes']):
            col1, col2 = st.columns([4, 1])
            with col1:
                edited = st.text_input(f"Process {i+1}:", value=process, key=f"process_{i}")
                if edited != process:
                    parsed['processes'][i] = edited
                    st.session_state.stage1_edited = True
            with col2:
                if st.button("🗑️", key=f"del_process_{i}"):
                    processes_to_remove.append(i)
                    st.session_state.stage1_edited = True
        
        for idx in sorted(processes_to_remove, reverse=True):
            parsed['processes'].pop(idx)
        if processes_to_remove:
            st.rerun()
        
        # Data Stores
        st.markdown("#### 💾 Data Stores")
        col1, col2 = st.columns([4, 1])
        with col1:
            new_store = st.text_input("Add new data store:", key="new_store")
        with col2:
            if st.button("➕ Add", key="add_store"):
                if new_store.strip():
                    parsed['data_stores'].append(new_store.strip())
                    st.session_state.stage1_edited = True
                    st.rerun()
        
        stores_to_remove = []
        for i, store in enumerate(parsed['data_stores']):
            col1, col2 = st.columns([4, 1])
            with col1:
                edited = st.text_input(f"Data Store {i+1}:", value=store, key=f"store_{i}")
                if edited != store:
                    parsed['data_stores'][i] = edited
                    st.session_state.stage1_edited = True
            with col2:
                if st.button("🗑️", key=f"del_store_{i}"):
                    stores_to_remove.append(i)
                    st.session_state.stage1_edited = True
        
        for idx in sorted(stores_to_remove, reverse=True):
            parsed['data_stores'].pop(idx)
        if stores_to_remove:
            st.rerun()
        
        # Data Flows
        st.markdown("#### ➡️ Data Flows")
        col1, col2 = st.columns([4, 1])
        with col1:
            new_flow = st.text_input("Add new data flow (e.g., 'From Patient to Process 1, Label: Prescription request'):", key="new_flow")
        with col2:
            if st.button("➕ Add", key="add_flow"):
                if new_flow.strip():
                    parsed['data_flows'].append(new_flow.strip())
                    st.session_state.stage1_edited = True
                    st.rerun()
        
        flows_to_remove = []
        for i, flow in enumerate(parsed['data_flows']):
            col1, col2 = st.columns([4, 1])
            with col1:
                edited = st.text_input(f"Flow {i+1}:", value=flow, key=f"flow_{i}")
                if edited != flow:
                    parsed['data_flows'][i] = edited
                    st.session_state.stage1_edited = True
            with col2:
                if st.button("🗑️", key=f"del_flow_{i}"):
                    flows_to_remove.append(i)
                    st.session_state.stage1_edited = True
        
        for idx in sorted(flows_to_remove, reverse=True):
            parsed['data_flows'].pop(idx)
        if flows_to_remove:
            st.rerun()
        
        # Save edited version
        if st.session_state.stage1_edited or st.button("💾 Finalize Edits", key="finalize_edits"):
            st.session_state.stage1_result = reconstruct_stage1_content(parsed)
            st.session_state.stage1_finalized = True
            st.success("✅ Edits saved! You can now proceed to Stage 2.")
    
    # Stage 2: Error Analysis
    st.markdown("---")
    st.markdown("### Stage 2: Check for Errors")

    stage2_disabled = 'stage1_result' not in st.session_state
    if stage2_disabled:
        st.info("👆 Complete and finalize Stage 1 edits to unlock Stage 2")

    if st.button("⚠️ Run Stage 2", key="stage2", disabled=stage2_disabled):
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
            st.warning("Run and finalize Stage 1 first!")
    
    # Stage 3: System Assessment
    st.markdown("---")
    st.markdown("### Stage 3: Assess System Modeling")

    stage3_disabled = 'stage2_result' not in st.session_state
    if stage3_disabled:
        st.info("👆 Complete Stages 1 & 2 to unlock Stage 3")

    if st.button("🎯 Run Stage 3", key="stage3", disabled=stage3_disabled):
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
    
    # Reset button
    st.markdown("---")
    if st.button("🔄 Reset All Stages"):
        for key in ['stage1_result', 'stage2_result', 'stage1_raw', 'stage1_parsed', 'stage1_edited', 'stage1_finalized']:
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
        - **You can edit these lists before proceeding!**
        
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