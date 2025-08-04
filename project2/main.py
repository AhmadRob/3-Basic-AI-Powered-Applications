import streamlit as st # for web app interface
import PyPDF2 # for PDF file handling
import os # for environment variable handling
import io # for in-memory file handling
from openai import OpenAI # for OpenAI API interaction
from dotenv import load_dotenv # for loading environment variables from .env file

load_dotenv() # Load environment variables from .env file

# Set up the Streamlit app configuration
st.set_page_config(page_title="CV critique", page_icon=":guardsman:", layout="centered")

# Main application code
st.title("CV Critique App")
st.markdown("Upload your CV and get a feedback on its content and structure based on best practices.")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Get OpenAI API key from environment variable

uploaded_file = st.file_uploader("Upload your CV (PDF or TXT format)", type=["pdf", "txt"])
job_role = st.text_input("Enter the job role you are applying for (optional)", placeholder="e.g., Software Engineer, Data Scientist")

analyze = st.button("Analyze CV")

# Function to extract text from PDF files
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
        return text.strip()

# Function to extract text from uploaded file (PDF or TXT)
def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        return extract_text_from_pdf(io.BytesIO(uploaded_file.read()))
    return uploaded_file.getvalue().decode("utf-8")

if analyze and uploaded_file:
    st.write("Analyzing your CV... Please wait.")
    try:
        file_content = extract_text_from_file(uploaded_file)
        if not file_content.strip():
            st.error("The uploaded file is empty or could not be read. Please upload a valid CV.")
            st.stop()
        
        prompt = f"""Please analyze this resume and provide constructive feedback. 
        Focus on the following aspects:
        1. Content clarity and impact and conciseness
        2. Skills presentation
        3. Experience descriptions
        4. Specific improvements for {job_role if job_role else 'general job applications'}
        
        Resume content:
        {file_content}
        
        Please provide your analysis in a clear, structured format with specific recommendations."""

        client = OpenAI(OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert resume reviewer with years of experience in HR and recruitment."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        st.markdown("### Feedback:")
        st.markdown(response.choices[0].message.content)
        
    except Exception as e:
        st.error(f"An error occurred while processing your CV: {str(e)}")