from dotenv import load_dotenv  
load_dotenv() 
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
import google.generativeai as genai
import os
from io import BytesIO
import sweetviz as sv

# ==============================
# 1) API Keys & Configuration
# ==============================
# We'll configure Gemini (PaLM) later when needed

# ==============================
# 2) Streamlit Page Settings
# ==============================
st.set_page_config(page_title="Advanced Data Sweeper", layout="wide")

# ==============================
# 3) Custom Dark-Mode Styling
# ==============================
st.markdown(
    """
    <style>
        .main {
            background-color: #121212;
        }
        .block-container {
            padding: 3rem 2rem;
            border-radius: 12px;
            background-color: #1e1e1e;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        }
        h1, h2, h3, h4, h5, h6 {
            color: #66c2ff;
        }
        .stButton>button {
            border: none;
            border-radius: 8px;
            background-color: #0078D7;
            color: white;
            padding: 0.75rem 1.5rem;
            font-size: 1rem;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4);
        }
        .stButton>button:hover {
            background-color: #005a9e;
            cursor: pointer;
        }
        .stDataFrame, .stTable {
            border-radius: 10px;
            overflow: hidden;
        }
        .css-1aumxhk, .css-18e3th9 {
            text-align: left;
            color: white;
        }
        .stRadio>label {
            font-weight: bold;
            color: white;
        }
        .stCheckbox>label {
            color: white;
        }
        .stDownloadButton>button {
            background-color: #28a745;
            color: white;
        }
        .stDownloadButton>button:hover {
            background-color: #218838;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ==============================
# 4) Main Title and Description
# ==============================
st.title("Advanced Data Sweeper")
st.write("Transform your files between CSV and Excel formats with built-in data cleaning, AI insights, and visualization.")

# ==============================
# 5) File Uploader
# ==============================
uploaded_files = st.file_uploader(
    "Upload your files (CSV or Excel):",
    type=["csv", "xlsx"],
    accept_multiple_files=True
)

# ==============================
# 6) Process Each Uploaded File
# ==============================
if uploaded_files:
    for file in uploaded_files:
        file_ext = os.path.splitext(file.name)[-1].lower()

        # Read file into a DataFrame
        if file_ext == ".csv":
            df = pd.read_csv(file)
        elif file_ext == ".xlsx":
            df = pd.read_excel(file)
        else:
            st.error(f"Unsupported file type: {file_ext}")
            continue

        # --------------------------------------
        # Sidebar Components: Ask the Data (Gemini)
        # --------------------------------------
        with st.sidebar:
            st.subheader("💬 Ask the Data")
            user_query = st.text_input("Ask me anything about the dataset...")

            # Only proceed if there's a query and df is defined
            if user_query and "df" in locals():
                try:
                    # Configure Gemini/PaLM with your API key
                    genai.configure(api_key=os.getenv("GEMINI_KEY"))
            
                    # Initialize the model (e.g., "gemini-pro" or another available model)
                    model = genai.GenerativeModel("v1beta")

                    # Build the prompt with context about the data
                    prompt = (
                        "You are a data assistant. Please provide your answers in clear English.\n\n"
                         f"Data columns: {df.columns}\n"
                         f"Sample data:\n{df.head(2).to_string(index=False)}\n\n"
                         f"User question: {user_query}"
                    )

                    # Generate the response
                    response = model.generate_content(prompt)

                    # Display the AI assistant's reply
                    st.write("AI Assistant says:", response.text)

                except Exception as e:
                    st.error(f"AI Error: {str(e)}")

        # -------------------------
        # AI Summary (Main Section)
        # -------------------------
        st.subheader(f"🤖 AI Insights for {file.name}")
        use_ai = st.checkbox("Generate automatic insights using OpenAI", key=f"ai_{file.name}")
        if use_ai:
            try:
                # Summarize basic info to feed the AI
                summary_text = (
                    f"Columns: {df.columns.tolist()}\n"
                    f"First 3 rows:\n{df.head(3).to_string(index=False)}"
                )
                client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a data analyst. Provide 3 key insights in English."
                        },
                        {
                            "role": "user",
                            "content": summary_text
                        }
                    ]
                )
                ai_answer = response.choices[0].message.content
                st.write("**AI Report 📝**")
                st.write(ai_answer)
            except Exception as e:
                st.error(f"AI Error: {str(e)}")

        # -------------------------
        # Data Preview
        # -------------------------
        st.write("🔍 **Preview the Head of the DataFrame**")
        st.dataframe(df.head())

        # -------------------------
        # Data Cleaning Options
        # -------------------------
        st.subheader("🛠️ Data Cleaning Options")
        if st.checkbox(f"Clean Data for {file.name}", key=f"clean_{file.name}"):
            col1, col2, col3, col4 = st.columns(4)

            # 1) Remove Duplicates
            with col1:
                if st.button(f"Remove Duplicates from {file.name}", key=f"dup_{file.name}"):
                    df.drop_duplicates(inplace=True)
                    st.write("Duplicates Removed! ✅")

            # 2) Fill Missing Values
            with col2:
                if st.button(f"Fill Missing Values for {file.name}", key=f"fill_{file.name}"):
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                    st.success("Missing Values Filled! 💉")

            # 3) Auto-Clean Columns (drop columns with > 50% null)
            with col3:
                if st.button("Auto-Clean Columns 🧹", key=f"auto_{file.name}"):
                    null_percent = df.isnull().mean()
                    bad_cols = null_percent[null_percent > 0.5].index
                    df.drop(columns=bad_cols, inplace=True)
                    st.success(f"{len(bad_cols)} Columns Removed! 🗑️")

            # 4) Fix Outliers
            with col4:
                if st.button("Fix Weird Values 🔧", key=f"weird_{file.name}"):
                    numeric_cols = df.select_dtypes(include='number').columns
                    for col in numeric_cols:
                        q1 = df[col].quantile(0.25)
                        q3 = df[col].quantile(0.75)
                        iqr = q3 - q1
                        lower = q1 - 1.5 * iqr
                        upper = q3 + 1.5 * iqr
                        df[col] = df[col].clip(lower=lower, upper=upper)
                    st.success("Outliers Fixed! 📏")

        # -------------------------
        # Select Columns to Convert
        # -------------------------
        st.subheader("🎯 Select Columns to Convert")
        columns = st.multiselect(
            f"Choose Columns for {file.name}",
            df.columns,
            default=df.columns,
            key=f"cols_{file.name}"
        )
        df = df[columns]  # Filter to the selected columns

        # -------------------------
        # Data Visualization
        # -------------------------
        st.subheader("📊 Data Visualization")
        if st.checkbox(f"Show Visualization for {file.name}", key=f"viz_{file.name}"):
            chart_type = st.selectbox(
                "Select Chart Type",
                ["Bar", "Line", "Scatter", "Histogram"],
                key=f"chart_{file.name}"
            )

            # Bar Chart
            if chart_type == "Bar":
                st.bar_chart(df.select_dtypes(include='number'))

            # Histogram
            elif chart_type == "Histogram":
                plt.close('all')
                fig = plt.figure(figsize=(10, 6))
                df.hist(ax=fig.gca())
                st.pyplot(fig)
                plt.close(fig)

            # Scatter Plot
            elif chart_type == "Scatter":
                x = st.selectbox("X-axis", df.columns, key=f"x_{file.name}")
                y = st.selectbox("Y-axis", df.columns, key=f"y_{file.name}")
                st.scatter_chart(df[[x, y]])

            # Line Chart
            elif chart_type == "Line":
                st.line_chart(df.select_dtypes(include='number'))

            # Gemini (PaLM) Trend Predictions
            if st.button("🔮 Predict Future Trends", key=f"trend_{file.name}"):
                try:
                    genai.configure(api_key=os.getenv("GEMINI_KEY"))
                    model = genai.GenerativeModel("v1beta")
                    response = model.generate_content(
                        f"Here is a sample of my dataset:\n{df.head(3).to_string(index=False)}\n"
                        "Please predict trends for the next 6 months in English."
                    )
                    st.write("**Gemini's Answer:**", response.text)
                except Exception as e:
                    st.error(f"Gemini Error: {str(e)}")

        # -------------------------
        # Full Data Report (Sweetviz)
        # -------------------------
        if st.button("📈 Full Data Report", key=f"report_{file.name}"):
            report = sv.analyze(df)
            report.show_html("report.html")
            with open("report.html", "r", encoding="utf-8") as f:
                html_code = f.read()
            st.components.v1.html(html_code, height=1000, scrolling=True)

        # -------------------------
        # File Conversion (CSV/Excel)
        # -------------------------
        st.subheader("🔄 Conversion Options")
        conversion_type = st.radio(
            f"Convert {file.name} to:",
            ["CSV", "Excel"],
            key=f"convert_{file.name}"
        )

        if st.button(f"Convert {file.name}", key=f"convert_btn_{file.name}"):
            buffer = BytesIO()
            if conversion_type == "CSV":
                df.to_csv(buffer, index=False)
                file_name = file.name.replace(file_ext, ".csv")
                mime_type = "text/csv"
            else:  # Excel
                df.to_excel(buffer, index=False, engine='openpyxl')
                file_name = file.name.replace(file_ext, ".xlsx")
                mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

            buffer.seek(0)
            st.download_button(
                label=f"⬇️ Download {file.name} as {conversion_type}",
                data=buffer,
                file_name=file_name,
                mime=mime_type
            )

# Final success message
st.success("🎉 All files processed successfully!")
