import streamlit as st
import pandas as pd
import json
import csv
from utils.data_generator import SyntheticDataGenerator
from utils.drift_detector import DriftDetector, display_html_report
from utils.data_generator_using_meta_info import DataGenerationUsingMetaInfo


# API Key Configuration
OPENAI_API_KEY = st.secrets['api_keys']["OPENAI_API_KEY"]

# Initialize Classes
data_gen = SyntheticDataGenerator(api_key=OPENAI_API_KEY)
drift_detector = DriftDetector()

# Streamlit UI
st.title("Synthetic Data Generation App")

# Tabs for User Flow
tab1, tab2 = st.tabs(["Reference Data Input", "Metadata Input"])

# Initialize session state variables if they do not exist
if "synthetic_data" not in st.session_state:
    st.session_state.synthetic_data = None
if "synthetic_texts" not in st.session_state:
    st.session_state.synthetic_texts = None

########################################################## SCENARIO 1: REFERENCE DATA INPUT ##########################################################
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
                    textual_data_drift_preset_report_path, textual_data_embeddings_countour_plots_path, textual_embeddings_drift_mmd_report_path = drift_detector.textual_data_drift_reports(st.session_state.reference_data,
                                                                                                                                                    st.session_state.synthetic_data, 
                                                                                                                                                    text_column)
                    st.success("Drift Detection Report Generated! Download below.")
                    st.markdown('Data Drift Preset Report:')
                    display_html_report(textual_data_drift_preset_report_path)
                    st.markdown('#')
                    st.markdown('Embeddings Drift Report:')
                    display_html_report(textual_embeddings_drift_mmd_report_path)
                    st.markdown('##')
                    st.image(textual_data_embeddings_countour_plots_path)
                    st.download_button("Download Report", textual_data_drift_preset_report_path)
                


########################################################## SCENARIO 2: PLACEHOLDER FOR METADATA-BASED GENERATION ##########################################################
with tab2:
    data_generator_using_meta_info = DataGenerationUsingMetaInfo(api_key=OPENAI_API_KEY)

    if 'metadata_schema' not in st.session_state:
        st.session_state.metadata_schema = None

    if 'final_schema' not in st.session_state:
        st.session_state.final_schema = None

    if 'field_ranges' not in st.session_state:
        st.session_state.field_ranges = {}

    if 'step' not in st.session_state:
        st.session_state.step = 'input_prompt'

    # Streamlit UI
    st.title("Metadata-Based Synthetic Data Generation")

    # Step 1: User Input for Dataset Description
    if st.session_state.step == 'input_prompt':
        user_prompt = st.text_area(
            "Describe your dataset requirements:",
            placeholder="e.g., I need employee data with employee_id, name, role, designation, and band."
        )

        if st.button("Generate Metadata Schema"):
            if user_prompt.strip() == "":
                st.warning("Please enter a dataset description.")
            else:
                with st.spinner("Generating Metadata Schema..."):
                    metadata_schema = data_generator_using_meta_info.get_metadata_from_llm(user_prompt)
                    if metadata_schema:
                        st.session_state.metadata_schema = metadata_schema
                        st.session_state.step = 'confirm_schema'
                        st.rerun()

    # Step 2: Schema Confirmation
    elif st.session_state.step == 'confirm_schema':
        st.success("Generated Metadata Schema:")
        st.json(st.session_state.metadata_schema)

        edited_schema = st.text_area(
            "Edit Metadata Schema (if needed):",
            value=json.dumps(st.session_state.metadata_schema, indent=4),
            height=300
        )

        if st.button("Confirm Schema"):
            st.session_state.final_schema = json.loads(edited_schema)
            st.session_state.step = 'set_ranges'
            st.rerun()

    # Step 3: Field Range Selection
    elif st.session_state.step == 'set_ranges':
        st.header("Set Field Ranges/Constraints")

        for field, props in st.session_state.final_schema.items():
            st.subheader(f"Field: {field}")
            field_type = props.get('type', 'string')

            if field_type == 'number' or field_type == 'integer':
                min_value = st.number_input(f"Minimum value for {field}", value=0.0, key=f"{field}_min")
                max_value = st.number_input(f"Maximum value for {field}", value=100.0, key=f"{field}_max")
                st.session_state.field_ranges[field] = {'min': min_value, 'max': max_value}
            
            elif field_type == 'string':
                placeholder = st.text_input(f"Placeholder/Example value for {field}", key=f"{field}_placeholder")
                st.session_state.field_ranges[field] = {'placeholder': placeholder}
            
            elif field_type == 'boolean':
                default_value = st.checkbox(f"Default value for {field}", key=f"{field}_default")
                st.session_state.field_ranges[field] = {'default': default_value}

            elif field_type == 'date':
                start_date = st.date_input(f"Start date for {field}", key=f"{field}_start_date")
                end_date = st.date_input(f"End date for {field}", key=f"{field}_end_date")
                st.session_state.field_ranges[field] = {'start_date': start_date, 'end_date': end_date}

            st.write("---")

        num_records = st.number_input("Number of records to generate", min_value=1, value=10, step=1, key='num_records')

        if st.button("Proceed to Data Generation"):
            # Use the widget value directly, no need to modify session state here
            st.session_state.step = 'generate_data'
            st.rerun()

    # Step 4: Generate and Display Data using LLM
    elif st.session_state.step == 'generate_data':
        st.header("Synthetic Data Preview")

        with st.spinner("Generating Synthetic Data..."):
            # Send data to LLM for synthetic data generation
            synthetic_data_response = data_generator_using_meta_info.generate_synthetic_data_llm(
                st.session_state.final_schema,
                st.session_state.field_ranges,
                st.session_state.num_records  # Use the num_records directly
            )

            synthetic_data_response_lines = synthetic_data_response.split("\n")
            csv_filename = "./outputs/synthetic_data/using_metadata/tabular/synthetic_data.csv"
            with open(csv_filename, mode="w", newline='') as file:
                writer = csv.writer(file)
                
                # Write the header (first line in the output)
                writer.writerow(synthetic_data_response_lines[0].split(","))
                
                # Write the records (remaining lines in the output)
                for line in synthetic_data_response_lines[1:]:
                    writer.writerow(line.split(","))

            # Load the CSV into a DataFrame using pandas
            synthetic_data_df = pd.read_csv(csv_filename)

            # Display the CSV data in Streamlit
            st.dataframe(synthetic_data_df)

            # Optionally, allow the user to download the generated CSV
            st.download_button(
                label="Download CSV",
                data=synthetic_data_df.to_csv(index=False),
                file_name=csv_filename,
                mime="text/csv"
            )

            print("CSV file generated and displayed successfully!")

        if st.button("Restart"):
            st.session_state.step = 'input_prompt'
            st.session_state.metadata_schema = None
            st.session_state.final_schema = None
            st.session_state.field_ranges = {}
            st.session_state.num_records = 0
            st.rerun()






























    # # Ensure session state keys exist
    # if 'metadata_schema' not in st.session_state:
    #     st.session_state.metadata_schema = None

    # if 'final_schema' not in st.session_state:
    #     st.session_state.final_schema = None

    # if 'field_ranges' not in st.session_state:
    #     st.session_state.field_ranges = {}

    # if 'step' not in st.session_state:
    #     st.session_state.step = 'input_prompt'

    # if 'num_records' not in st.session_state: 
    #     st.session_state.num_records = 10  
    #     st.session_state.step = 'input_prompt'

    # # Streamlit UI
    # st.title("Metadata-Based Synthetic Data Generation")

    # # Step 1: User Input for Dataset Description
    # if st.session_state.step == 'input_prompt':
    #     user_prompt = st.text_area(
    #         "Describe your dataset requirements:",
    #         placeholder="e.g., I need employee data with employee_id, name, role, designation, and band."
    #     )

    #     if st.button("Generate Metadata Schema"):
    #         if user_prompt.strip() == "":
    #             st.warning("Please enter a dataset description.")
    #         else:
    #             with st.spinner("Generating Metadata Schema..."):
    #                 metadata_schema = get_metadata_from_llm(user_prompt)
    #                 if metadata_schema:
    #                     st.session_state.metadata_schema = metadata_schema
    #                     st.session_state.step = 'confirm_schema'
    #                     st.rerun()

    # # Step 2: Schema Confirmation
    # elif st.session_state.step == 'confirm_schema':
    #     st.success("Generated Metadata Schema:")
    #     st.json(st.session_state.metadata_schema)

    #     edited_schema = st.text_area(
    #         "Edit Metadata Schema (if needed):",
    #         value=json.dumps(st.session_state.metadata_schema, indent=4),
    #         height=300
    #     )

    #     if st.button("Confirm Schema"):
    #         st.session_state.final_schema = json.loads(edited_schema)
    #         st.session_state.step = 'set_ranges'
    #         st.rerun()

    # # Step 3: Field Range Selection
    # elif st.session_state.step == 'set_ranges':
    #     st.header("Set Field Ranges/Constraints")

    #     for field, props in st.session_state.final_schema.items():
    #         st.subheader(f"Field: {field}")
    #         field_type = props.get('type', 'string')

    #         if field_type == 'number' or field_type == 'integer':
    #             min_value = st.number_input(f"Minimum value for {field}", value=0.0, key=f"{field}_min")
    #             max_value = st.number_input(f"Maximum value for {field}", value=100.0, key=f"{field}_max")
    #             st.session_state.field_ranges[field] = {'min': min_value, 'max': max_value}
            
    #         elif field_type == 'string':
    #             placeholder = st.text_input(f"Placeholder/Example value for {field}", key=f"{field}_placeholder")
    #             st.session_state.field_ranges[field] = {'placeholder': placeholder}
            
    #         elif field_type == 'boolean':
    #             default_value = st.checkbox(f"Default value for {field}", key=f"{field}_default")
    #             st.session_state.field_ranges[field] = {'default': default_value}

    #         elif field_type == 'date':
    #             start_date = st.date_input(f"Start date for {field}", key=f"{field}_start_date")
    #             end_date = st.date_input(f"End date for {field}", key=f"{field}_end_date")
    #             st.session_state.field_ranges[field] = {'start_date': start_date, 'end_date': end_date}

    #         st.write("---")

    #     num_records = st.number_input("Number of records to generate", min_value=1, value=10, step=1, key='num_records')

    #     if st.button("Proceed to Data Generation"):
    #         st.session_state.num_records = num_records
    #         st.session_state.step = 'generate_data'
    #         st.rerun()

    # # Step 4: Generate and Display Data using LLM
    # elif st.session_state.step == 'generate_data':
    #     st.header("Synthetic Data Preview")

    #     with st.spinner("Generating Synthetic Data..."):
    #         # Send data to LLM for synthetic data generation
    #         synthetic_data_csv = generate_synthetic_data_llm(
    #             st.session_state.final_schema,
    #             st.session_state.field_ranges,
    #             st.session_state.num_records
    #         )

    #         # Display CSV as DataFrame (first 5 rows)
    #         from io import StringIO
    #         synthetic_data = StringIO(synthetic_data_csv)
    #         df = pd.read_csv(synthetic_data)
    #         st.dataframe(df)

    #         # Provide download option
    #         csv = df.to_csv(index=False).encode('utf-8')
    #         st.download_button(
    #             label="Download Data as CSV",
    #             data=csv,
    #             file_name="synthetic_data.csv",
    #             mime="text/csv"
    #         )

    #     if st.button("Restart"):
    #         st.session_state.step = 'input_prompt'
    #         st.session_state.metadata_schema = None
    #         st.session_state.final_schema = None
    #         st.session_state.field_ranges = {}
    #         st.session_state.num_records = 0
    #         st.rerun()

    
