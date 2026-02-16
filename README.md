# Supabase Production Analytics Bot

This is a Streamlit application that uses LangChain, OpenAI, and Supabase to analyze and visualize production data.

## Prerequisites

- [Python 3.8+](https://www.python.org/downloads/)
- pip (Python package manager)

## 1. Install Dependencies

It is recommended to use a virtual environment to manage dependencies locally.

```bash
# Create virtual environment relative to the project directory
python -m venv venv

# If on macOS/Linux:
source venv/bin/activate

# If on Windows:
# venv\Scripts\activate

# Install the required packages
pip install -r requirements.txt
```

## 2. Configure Environment Variables

Create a file named `.env` in the root of the project directory and add the following keys. You will need to fill in your specific credentials.

```ini
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT="your_azure_endpoint_here"
AZURE_OPENAI_API_KEY="your_azure_api_key_here"
AZURE_OPENAI_API_VERSION="your_api_version_here"     # e.g., 2023-05-15
AZURE_OPENAI_DEPLOYMENT_NAME="your_deployment_name"

# Supabase Configuration
SUPABASE_URL="your_supabase_url"
SUPABASE_ANON_KEY="your_supabase_anon_key"
SUPABASE_DB_PASSWORD="your_database_password"
```

## 3. Run the Application

Execute the following command to start the Streamlit app:

```bash
streamlit run app.py
```

The application should open automatically in your default web browser (usually at http://localhost:8501).
