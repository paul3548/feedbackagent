# CLAUDE.md — feedbackagent

This file provides guidance for AI assistants working in this repository.

## Project Overview

`feedbackagent` is an educational Streamlit web application that gives students
automated feedback on Data Flow Diagrams (DFDs). Students upload a DFD image
(PNG/JPG) and the app runs a three-stage OpenAI vision analysis, producing
structured feedback on notation correctness, error detection, and how well the
diagram models a target system.

The default target system is the **NHS electronic prescribing system**, but the
scenario is editable by the user at runtime.

## Repository Structure

```
feedbackagent/
├── feedback.py              # Main Streamlit app — the entry point
├── staged_dfd_analyzer.py   # Analyzer classes used by feedback.py
├── dfd_analyzer.py          # Original single-stage analyzer (legacy/reference)
├── requirements.txt         # Python dependencies
├── .gitignore
└── README.md
```

## Running the Application

```bash
pip install -r requirements.txt
streamlit run feedback.py
```

The app will open at `http://localhost:8501`.

## Configuration

### OpenAI API Key

The app reads the key from Streamlit secrets first, then falls back to a
password text input in the UI.

To configure via secrets, create `.streamlit/secrets.toml` (gitignored):

```toml
OPENAI_API_KEY = "sk-..."
```

### Model Selection

`feedback.py` has a top-level constant:

```python
MODEL_CHOICE = "gpt-5"  # line 7
```

Change this to `"gpt-4o"`, `"gpt-4o-mini"`, or `"gpt-4-turbo"` to use a
different model. The model name is passed through to every OpenAI API call.
`gpt-5` is the default and is the most capable option when available.

## Key Files in Detail

### `feedback.py` — Main Streamlit Application

**Entry point.** Imports `SelectiveCacheDFDAnalyzer` from
`staged_dfd_analyzer.py`. All Streamlit session state and UI logic lives here.

**Top-level helpers:**

- `parse_stage1_result(content)` — parses the structured Stage 1 markdown
  response into a dict with keys `notation_check`, `external_entities`,
  `processes`, `data_stores`, `data_flows`.
- `reconstruct_stage1_content(parsed_data)` — serialises the edited dict back
  into the expected markdown format that Stage 2 and 3 prompts consume.

**UI flow:**

1. Expandable "System Scenario" section — editable text area. Calls
   `analyzer.update_system_description()` on save.
2. Analyzer is initialised once into `st.session_state.analyzer`.
3. File uploader for the DFD image.
4. Three sequential stage buttons. Stage 2 is disabled until
   `st.session_state.stage1_result` exists; Stage 3 is disabled until
   `st.session_state.stage2_result` exists.
5. Between Stage 1 and Stage 2 there is an editable section where the user can
   add/edit/delete the parsed component lists before finalising.
6. "Reset All Stages" button clears all stage keys from session state.

**Session state keys used:**

| Key | Set by | Purpose |
|-----|--------|---------|
| `analyzer` | init block | The `SelectiveCacheDFDAnalyzer` instance |
| `current_scenario` | scenario section | Active scenario text |
| `stage1_raw` | Stage 1 button | Raw LLM response string |
| `stage1_parsed` | Stage 1 button | Parsed dict from `parse_stage1_result` |
| `stage1_edited` | Edit section | Boolean flag, marks unsaved edits |
| `stage1_finalized` | Finalize button | Boolean, unlocks Stage 2 |
| `stage1_result` | Finalize button | Reconstructed content string for Stage 2/3 |
| `stage2_result` | Stage 2 button | Stage 2 LLM response string |

### `staged_dfd_analyzer.py` — Analyzer Classes

Contains two classes:

#### `SelectiveCacheDFDAnalyzer` (active — used by `feedback.py`)

- Caches only the static prompt templates in `self.prompt_cache` (a plain
  dict). Does **not** cache API results — every student gets a fresh analysis.
- `stage1_analyze_notation(image, model)` — encodes image to base64 JPEG,
  calls OpenAI with vision, returns dict `{success, content, tokens_used,
  model}`.
- `stage2_check_errors(description, model)` — text-only call; formats
  `stage2_prompt_template` with the description string.
- `stage3_assess_system(description, error_analysis, model)` — text-only call;
  formats `stage3_prompt_template` with description, error_analysis, and
  `self.system_description`.
- `update_system_description(new_description)` — updates `self.system_description`
  and re-caches prompts.
- `analyze_complete(image, model)` — convenience method that runs all three
  stages sequentially and aggregates token counts.
- `_call_openai(prompt, image_base64, model)` — low-level wrapper; always uses
  `max_completion_tokens=8000`.

#### `StagedDFDAnalyzer` (legacy — not imported by `feedback.py`)

Similar API but adds MD5-based result caching (`self.image_cache`). Useful if
the same image is submitted multiple times in a session. Not currently wired
into the main UI.

### `dfd_analyzer.py` — Original Analyzer (legacy/reference)

`DFDAnalyzer` class with a single-pass approach. Prompt is composed from four
modular components: `task_description`, `system_description`, `common_faults`,
`feedback_guidance`. Has both async (`analyze_dfd`) and sync
(`analyze_dfd_sync`) methods. Not used by `feedback.py`; kept for reference
and as a simpler integration example.

## Prompt Architecture

All three stage prompts expect the LLM to respond with specific **bold
headers** that the parser relies on:

- `**NOTATION CHECK**`
- `**EXTERNAL ENTITIES**`
- `**PROCESSES**`
- `**DATA STORES**`
- `**DATA FLOWS**`

Items under list sections must start with `- ` to be parsed correctly by
`parse_stage1_result`. Changing the header names or list prefix format in the
prompts will break parsing.

Stage 2 and Stage 3 prompts use Python `.format()` with `{description}`,
`{error_analysis}`, and `{system_description}` placeholders. Do not use
f-strings or other formatting approaches; curly braces not used as placeholders
must not appear in the templates.

## Dependencies

```
streamlit     — web UI framework
openai        — Python SDK for OpenAI API (synchronous client only)
Pillow        — image loading and base64 encoding
```

No pinned versions in `requirements.txt`. Images are converted to JPEG at
quality 95 before being base64-encoded for the API.

## Known Issues / Notes

- Line 361 in `staged_dfd_analyzer.py` contains a stray `e` character
  (`e        ` at the start of a line inside the `StagedDFDAnalyzer.stage2_prompt`
  string). This is a typo in a prompt string — harmless but worth cleaning up.
- `SelectiveCacheDFDAnalyzer.stage2_check_errors` contains a hardcoded
  `st.info("⚠️ Checking errors")` call (line 212 of `staged_dfd_analyzer.py`),
  which couples the analyzer class to Streamlit. Other equivalent info calls in
  the same class are commented out. Be aware when refactoring.
- `StagedDFDAnalyzer` (`stage2_check_errors`, line 510) formats
  `self.stage2_prompt` with `system_description=...` but the template has no
  `{system_description}` placeholder — this extra kwarg is silently ignored by
  Python's `str.format()`.
- There are no automated tests. Testing is done manually through the Streamlit
  UI.
- There is no CI/CD configuration.

## Development Conventions

- **No test suite**: changes are validated by running the app manually.
- **Model names**: prefer `"gpt-4o"` as a safe default in new code; `"gpt-5"`
  is used in the current UI constant.
- **Session state**: always check for key existence before reading
  (`'key' in st.session_state`) to avoid `KeyError`.
- **Analyzer class**: use `SelectiveCacheDFDAnalyzer`. Do not use
  `StagedDFDAnalyzer` or `DFDAnalyzer` for new work without explicit reason.
- **Prompt changes**: when editing Stage 1 prompt headers, update
  `parse_stage1_result` in `feedback.py` to match.
- **API responses**: all `_call_openai` calls return `{"success": bool, ...}`;
  always check `result["success"]` before accessing `result["content"]`.
- **Image handling**: images are accepted as PIL `Image` objects and converted
  to JPEG internally. Do not pass raw bytes or file paths to analyzer methods.

## Deployment

The app is intended to be deployed via Streamlit Community Cloud or any host
that supports `streamlit run`. The `.streamlit/secrets.toml` file must be
configured with `OPENAI_API_KEY` in production.
