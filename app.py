import streamlit as st
import pandas as pd
from utils.data_generator import SyntheticDataGenerator
from utils.drift_detector import DriftDetector, display_html_report

# API Key Configuration
OPENAI_API_KEY = st.secrets['api_keys']["OPENAI_API_KEY"]

# Initialize Classes
data_gen = SyntheticDataGenerator(api_key=OPENAI_API_KEY)
drift_detector = DriftDetector()

# Streamlit UI
st.title("Synthetic Data Generation App")

# Tabs for User Flow
tab1, tab2 = st.tabs(["Reference Data Input", "Metadata Input (Coming Soon)"])

# Initialize session state variables if they do not exist
if "synthetic_data" not in st.session_state:
    st.session_state.synthetic_data = None
if "synthetic_texts" not in st.session_state:
    st.session_state.synthetic_texts = None

# Scenario 1: Reference Data Input
with tab1:
    st.header("Generate Synthetic Data from Reference Data")
    data_type = st.selectbox("Select Data Type", ["Tabular", "Textual"])

    if data_type == "Tabular":
        uploaded_file = st.file_uploader("Upload Reference CSV", type=["csv"])

        if uploaded_file:
            reference_data = pd.read_csv(uploaded_file)
            st.write("Reference Data Preview:", reference_data.head())

            num_rows = st.number_input("Number of Synthetic Rows", min_value=10, value=100)
            if st.button("Generate Synthetic Tabular Data"):
                st.session_state.synthetic_data = data_gen.generate_tabular_data(reference_data, num_rows)
                st.write("Synthetic Data Preview:", st.session_state.synthetic_data.head())

        if st.session_state.synthetic_data is not None:
            if st.button("Run Drift Detection"):
                drift_report = drift_detector.detect_tabular_drift(reference_data, st.session_state.synthetic_data)
                drift_report.save_html("tabular_drift_report.html")
                st.success("Drift Detection Report Generated! Download below.")
                st.download_button("Download Report", "tabular_drift_report.html")

    elif data_type == "Textual":
        uploaded_file = st.file_uploader("Upload Reference CSV for Text Data", type=["csv"])
        reference_data = None
        synthetic_data = None
        if uploaded_file is not None:
            try:
                reference_data = pd.read_csv(uploaded_file)
                if reference_data.empty:
                    st.error("The uploaded CSV file is empty. Please upload a valid CSV with data.")
                else:
                    columns = reference_data.columns.tolist()
                    if not columns:
                        st.error("No columns found in the uploaded file. Please check your CSV.")
                    else:
                        text_column = st.selectbox(
                            "Select the text data column you want to augment data for", 
                            tuple(columns)
                        )
            except pd.errors.EmptyDataError:
                st.error("The uploaded file is empty. Please upload a non-empty CSV file.")
            except pd.errors.ParserError:
                st.error("Failed to parse the CSV file. Ensure it is formatted correctly.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")        
            
            if text_column:
                st.session_state.reference_data = reference_data
                st.dataframe(reference_data)

                if text_column not in reference_data.columns:
                    st.error(f"Column '{text_column}' not found in the uploaded CSV.")
                else:
                    reference_texts = reference_data[text_column].dropna().tolist()
                    st.write("Sample Reference Texts:", reference_texts[:5])

                    num_samples = st.number_input("Number of Synthetic Text Samples", min_value=1, value=5)
                    if st.button("Generate Synthetic Textual Data"):
                        st.session_state.synthetic_texts = data_gen.generate_textual_data("\n".join(reference_texts), num_samples)
                        synthetic_data = pd.DataFrame(st.session_state.synthetic_texts, columns=[text_column])
                        synthetic_data.dropna(inplace=True)
                        synthetic_data.reset_index(inplace=True, drop=True)
                        st.session_state.synthetic_data = synthetic_data
                        st.write("Synthetic Textual Data:", synthetic_data)

            if st.session_state.synthetic_data is not None:
                if st.button("Run Drift Detection"):
                    textual_data_drift_preset_report_path, textual_data_embeddings_countour_plots_path = drift_detector.textual_data_drift_reports(st.session_state.reference_data,
                                                                                                                                                    st.session_state.synthetic_data, 
                                                                                                                                                    text_column)
                    st.success("Drift Detection Report Generated! Download below.")
                    display_html_report(textual_data_drift_preset_report_path)
                    st.markdown('##')
                    st.image(textual_data_embeddings_countour_plots_path)
                    st.download_button("Download Report", textual_data_drift_preset_report_path)
                


# Scenario 2: Placeholder for Metadata-Based Generation
with tab2:
    st.header("Generate Synthetic Data from Metadata (Coming Soon)")
    st.info("This feature will be implemented in the next phase.")
