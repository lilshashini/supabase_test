#from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import logging
import sys
from datetime import datetime
import numpy as np
import os 
from supabase import create_client, Client
from sqlalchemy import create_engine
import psycopg2
from urllib.parse import quote_plus
from sqlalchemy import text  # Add this import
from dotenv import load_dotenv


load_dotenv()

def get_env_var(key, default=None):
    """Get environment variable from Streamlit secrets or OS"""
    try:
        # For Streamlit Cloud - use st.secrets
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except Exception:
        # If secrets don't exist, fall back to environment variables
        pass
    
    # For local development - use os.getenv
    return os.getenv(key, default)

# Configure logging
def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('chatbot.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def init_supabase_database(supabase_url: str, supabase_key: str, db_password: str) -> SQLDatabase:
    """Initialize Supabase database connection with logging"""
    try:
        # Extract connection details from Supabase URL
        # Supabase URL format: https://your-project-id.supabase.co
        project_id = supabase_url.replace('https://', '').replace('.supabase.co', '')
        
       # Pooler connection info
        host = "aws-0-ap-southeast-1.pooler.supabase.com"
        port = 6543
        database = "postgres"
        user = f"postgres.{project_id}"

        # Encode password for special characters
        encoded_password = quote_plus(db_password)

        # PostgreSQL connection string (Transaction Pooler)
        db_uri = f"postgresql://{user}:{encoded_password}@{host}:{port}/{database}"

        logger.info(f"Attempting to connect to Supabase transaction pooler: {host}:{port}/{database}")
        
        # Test connection first
        engine = create_engine(db_uri)
        connection = engine.connect()
        connection.close()
        
        # Create SQLDatabase instance
        db = SQLDatabase.from_uri(db_uri)
        logger.info("Supabase database connection successful")
        return db
        
    except Exception as e:
        logger.error(f"Supabase database connection failed: {str(e)}")
        raise e

def init_supabase_client(supabase_url: str, supabase_key: str) -> Client:
    """Initialize Supabase client for additional operations"""
    try:
        supabase_client = create_client(supabase_url, supabase_key)
        logger.info("Supabase client initialized successfully")
        return supabase_client
    except Exception as e:
        logger.error(f"Supabase client initialization failed: {str(e)}")
        raise e





def detect_visualization_request(user_query: str):
    """Enhanced visualization detection with better single and multi-machine support"""
    user_query_lower = user_query.lower()
    logger.info(f"Analyzing query for visualization: {user_query}")
    
    # Keywords that indicate visualization request
    viz_keywords = [
        'plot', 'chart', 'graph', 'visualize', 'show', 'display',
        'bar chart', 'line chart', 'pie chart', 'histogram', 'scatter plot',
        'bar', 'line', 'pie', 'trend', 'comparison', 'grouped bar', 'stacked bar',
        'pulse', 'pulse per minute', 'pulse rate', 'pulse variation', 'pulse trend'
    ]
    
    needs_viz = any(keyword in user_query_lower for keyword in viz_keywords)
    
    # Enhanced chart type detection with single machine support
    chart_type = "bar"  # default
    
    # Check for multi-machine/multi-category requests
    multi_machine_keywords = ['all machines', 'each machine', 'by machine', 'machines production', 'three machines']
    multi_category_keywords = ['by day', 'daily', 'monthly', 'by month','each day', 'each month']
    
    is_multi_machine = any(keyword in user_query_lower for keyword in multi_machine_keywords)
    is_multi_category = any(keyword in user_query_lower for keyword in multi_category_keywords)
    
    # Determine chart type based on query content
    if is_multi_machine and (is_multi_category or 'bar' in user_query_lower):
        chart_type = "multi_machine_bar"
    elif any(word in user_query_lower for word in ['line', 'trend', 'over time', 'hourly', 'daily', 'time series']):
        chart_type = "line"
    elif any(word in user_query_lower for word in ['pie', 'proportion', 'percentage', 'share', 'distribution']):
        chart_type = "pie"
    elif any(word in user_query_lower for word in ['scatter', 'relationship', 'correlation']):
        chart_type = "scatter"
    elif any(word in user_query_lower for word in ['histogram', 'distribution', 'frequency']):
        chart_type = "histogram"
    elif any(word in user_query_lower for word in ['grouped', 'stacked', 'multiple']):
        chart_type = "grouped_bar"
    elif any(word in user_query_lower for word in ['pulse', 'pulse per minute', 'rate per minute']):
        chart_type = "pulse_line"  # New chart type for pulse data
        
    logger.info(f"Visualization needed: {needs_viz}, Chart type: {chart_type}, Multi-machine: {is_multi_machine}")
    return needs_viz, chart_type

def get_enhanced_sql_chain(db):
    """Enhanced SQL chain with PostgreSQL-specific syntax"""
    template = """
    You are an expert data analyst. Based on the table schema below, write a PostgreSQL query that answers the user's question.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    IMPORTANT GUIDELINES FOR POSTGRESQL:
    
    1. Use PostgreSQL-specific functions and syntax:
       - Use EXTRACT() instead of YEAR(), MONTH(): EXTRACT(YEAR FROM actual_start_time)
       - Use DATE_TRUNC() for grouping: DATE_TRUNC('day', actual_start_time)
       - Use TO_CHAR() for date formatting: TO_CHAR(actual_start_time, 'YYYY-MM-DD')
       - Use CURRENT_DATE instead of NOW() for date comparisons
       - Use INTERVAL for date arithmetic: actual_start_time >= CURRENT_DATE - INTERVAL '7 days'
    
    2. For multi-machine production queries by time period:
       - Always include device_name/machine_name as a separate column
       - Use DATE_TRUNC('day', actual_start_time) for daily grouping
       - Use SUM() for production values
       - GROUP BY both time period AND machine/device
       - ORDER BY time period, then machine name
    
    3. For machine efficiency queries:
       - Calculate efficiency as a percentage: (actual_output / NULLIF(target_output, 0)) * 100 AS efficiency
       - Use NULLIF to avoid division by zero
       - Include machine/device name
       - Filter for specific dates if mentioned
       - ORDER BY efficiency DESC or machine name
    
    4. For April 2025 data specifically:
       - Use WHERE clause: WHERE DATE_TRUNC('month', actual_start_time) = '2025-04-01'::date
       - Or use: WHERE EXTRACT(YEAR FROM actual_start_time) = 2025 AND EXTRACT(MONTH FROM actual_start_time) = 4
    
    5. For daily data:
       - Use DATE_TRUNC('day', actual_start_time) for grouping
       - Alias as 'production_date' or 'day'
    
    6. Column naming conventions:
       - Use clear aliases: production_output AS daily_production
       - Use device_name AS machine_name for consistency
       - Use DATE_TRUNC('day', actual_start_time) AS production_date
       - Use efficiency AS efficiency_percent
    
    7. Data quality:
       - Handle NULL values: WHERE production_output IS NOT NULL
       - Filter out zero values if needed: AND production_output > 0
       
    8. For pulse per minute calculations:
        - Use LAG() function: LAG(length) OVER (PARTITION BY device_name ORDER BY timestamp)
        - Calculate pulse as: length - LAG(length) OVER (PARTITION BY device_name ORDER BY timestamp) AS pulse_per_minute
        - Filter out NULL values from LAG calculation
        - Order by timestamp for proper sequence
    
    POSTGRESQL EXAMPLES:
    
    Question: "Show all three machines production by each machine in April with bar chart for all 30 day"
    SQL Query: SELECT 
        DATE_TRUNC('day', actual_start_time)::date AS production_date,
        device_name AS machine_name,
        SUM(production_output) AS daily_production
    FROM hourly_production 
    WHERE EXTRACT(YEAR FROM actual_start_time) = 2024 
        AND EXTRACT(MONTH FROM actual_start_time) = 4 
        AND production_output IS NOT NULL 
        AND production_output > 0
    GROUP BY DATE_TRUNC('day', actual_start_time), device_name
    ORDER BY production_date, machine_name
    
    Question: "Plot the bar chart showing each machine's efficiency on April 1 2025"
    SQL Query: SELECT 
        device_name AS machine_name,
        ROUND((SUM(production_output) / NULLIF(SUM(target_output), 0)) * 100, 2) AS efficiency_percent
    FROM hourly_production 
    WHERE DATE_TRUNC('day', actual_start_time) = '2025-04-01'::date
        AND production_output IS NOT NULL 
        AND target_output IS NOT NULL
        AND target_output > 0
    GROUP BY device_name
    ORDER BY efficiency_percent DESC
    
    Question: "give the bar chart showing each machine's production on April 1 2025"
    SQL Query: SELECT 
        device_name AS machine_name,
        SUM(production_output) AS daily_production
    FROM daily_production_1
    WHERE DATE_TRUNC('day', actual_start_time) = '2025-04-01'::date
        AND production_output IS NOT NULL
        AND production_output > 0
    GROUP BY device_name
    ORDER BY machine_name

    Question: "Show machine efficiency for each machine in April 2025"
    SQL Query: SELECT 
        device_name AS machine_name,
        ROUND(AVG((production_output / NULLIF(target_output, 0)) * 100), 2) AS efficiency_percent
    FROM hourly_production 
    WHERE EXTRACT(YEAR FROM actual_start_time) = 2025 
        AND EXTRACT(MONTH FROM actual_start_time) = 4
        AND production_output IS NOT NULL 
        AND target_output IS NOT NULL
        AND target_output > 0
    GROUP BY device_name
    ORDER BY efficiency_percent DESC
    
    Question: "Compare production by machine for last 7 days"
    SQL Query: SELECT 
        DATE_TRUNC('day', actual_start_time)::date AS production_date,
        device_name AS machine_name,
        SUM(production_output) AS daily_production
    FROM hourly_production 
    WHERE actual_start_time >= CURRENT_DATE - INTERVAL '7 days'
        AND production_output IS NOT NULL
    GROUP BY DATE_TRUNC('day', actual_start_time), device_name
    ORDER BY production_date, machine_name
    
    Question: "Show pulse per minute for Machine1 on June 1st"
    SQL Query: SELECT 
        device_name,
        timestamp,
        length,
        length - LAG(length) OVER (PARTITION BY device_name ORDER BY timestamp) AS pulse_per_minute
    FROM length_data 
    WHERE DATE_TRUNC('day', timestamp) = '2025-06-01'::date 
        AND device_name = 'Machine1'
        AND length IS NOT NULL
    ORDER BY timestamp
    
    Write only the SQL query and nothing else. Do not wrap it in backticks or other formatting.
    
    Question: {question}
    SQL Query:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Fixed Azure OpenAI configuration
    llm = AzureChatOpenAI(
        azure_endpoint=get_env_var("AZURE_OPENAI_ENDPOINT"),
        api_key=get_env_var("AZURE_OPENAI_API_KEY"),
        api_version=get_env_var("AZURE_OPENAI_API_VERSION"),
        azure_deployment=get_env_var("AZURE_OPENAI_DEPLOYMENT_NAME"),
        temperature=0,
        max_tokens=800,
    )
    
    def get_schema(_):
        return db.get_table_info()
    
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def create_enhanced_visualization(df, chart_type, user_query):
    """Enhanced visualization with proper single and multi-machine support"""
    try:
        logger.info(f"Creating visualization: {chart_type}")
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"DataFrame columns: {list(df.columns)}")
        logger.info(f"DataFrame sample data:\n{df.head()}")
        
        if df.empty:
            logger.warning("DataFrame is empty")
            st.warning("No data available for visualization")
            return False
        
        # Clean and prepare data
        df = df.dropna()
        if df.empty:
            logger.warning("DataFrame is empty after removing NaN values")
            st.warning("No valid data available after cleaning")
            return False
        
        # Identify column types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Handle datetime columns
        for col in df.columns:
            if df[col].dtype == 'object' and col.lower() in ['production_date', 'day', 'date', 'production_month', 'month']:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    logger.info(f"Converted column {col} to datetime")
                except:
                    pass
        
        logger.info(f"Numeric columns: {numeric_cols}")
        logger.info(f"Categorical columns: {categorical_cols}")
        
        fig = None
        
        # Enhanced multi-machine bar chart
        if chart_type == "multi_machine_bar" or chart_type == "grouped_bar":
            # Look for standard column patterns
            date_col = None
            machine_col = None
            value_col = None
            
            # Find date column
            for col in df.columns:
                if col.lower() in ['production_date', 'date', 'day', 'production_month', 'month', 'hour']:
                    date_col = col
                    break
            
            # Find machine column
            for col in df.columns:
                if col.lower() in ['machine_name', 'device_name', 'machine', 'device']:
                    machine_col = col
                    break
            
            # Find value column
            for col in df.columns:
                if col.lower() in ['daily_production', 'monthly_production', 'production_output', 'total_output', 'production']:
                    value_col = col
                    break
                elif col in numeric_cols:
                    value_col = col
                    break
            
            logger.info(f"Detected columns - Date: {date_col}, Machine: {machine_col}, Value: {value_col}")
            
            if date_col and machine_col and value_col:
                # Create grouped bar chart
                fig = px.bar(
                    df, 
                    x=date_col, 
                    y=value_col, 
                    color=machine_col,
                    title=f"Production by Machine and Date",
                    labels={
                        date_col: "Date",
                        value_col: "Production Output",
                        machine_col: "Machine"
                    },
                    barmode='group'
                )
                
                # Enhance the chart appearance
                fig.update_layout(
                    showlegend=True,
                    height=600,
                    margin=dict(l=50, r=50, t=80, b=50),
                    xaxis_title_font_size=14,
                    yaxis_title_font_size=14,
                    title_font_size=16,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                # Improve x-axis for dates
                if df[date_col].dtype == 'datetime64[ns]':
                    fig.update_xaxes(
                        tickangle=45,
                        tickformat='%Y-%m-%d'
                    )
                
            else:
                # Fallback to regular grouped bar chart
                if len(categorical_cols) >= 2 and len(numeric_cols) >= 1:
                    x_col = categorical_cols[0]
                    color_col = categorical_cols[1]
                    y_col = numeric_cols[0]
                    fig = px.bar(df, x=x_col, y=y_col, color=color_col,
                               title=f"Grouped Bar Chart: {y_col} by {x_col} and {color_col}",
                               barmode='group')
        
        # Other chart types (keeping existing logic)
        elif chart_type == "line":
            if len(df.columns) >= 2:
                x_col = df.columns[0]
                y_col = numeric_cols[0] if numeric_cols else df.columns[1]
                color_col = None
                
                # Check if we have machine data for multi-line chart
                if 'machine_name' in df.columns or 'device_name' in df.columns:
                    color_col = 'machine_name' if 'machine_name' in df.columns else 'device_name'
                
                fig = px.line(df, x=x_col, y=y_col, color=color_col,
                             title=f"Line Chart: {y_col} over {x_col}")
                
        elif chart_type == "pie":
            if len(df.columns) >= 2:
                labels_col = categorical_cols[0] if categorical_cols else df.columns[0]
                values_col = numeric_cols[0] if numeric_cols else df.columns[1]
                fig = px.pie(df, names=labels_col, values=values_col, 
                           title=f"Pie Chart: {values_col} by {labels_col}")
                
        elif chart_type == "bar":
            if len(df.columns) >= 2:
                x_col = categorical_cols[0] if categorical_cols else df.columns[0]
                y_col = numeric_cols[0] if numeric_cols else df.columns[1]
                fig = px.bar(df, x=x_col, y=y_col, 
                           title=f"Bar Chart: {y_col} by {x_col}")
        
        elif chart_type == "pulse_line":
            # Look for pulse-specific columns
            time_col = None
            pulse_col = None
            machine_col = None
    
            # Find timestamp column
            for col in df.columns:
                if col.lower() in ['timestamp', 'time', 'datetime']:
                    time_col = col
                    break
    
            # Find pulse column
            for col in df.columns:
                if col.lower() in ['pulse_per_minute', 'pulse', 'pulse_rate']:
                    pulse_col = col
                    break
    
            # Find machine column
            for col in df.columns:
                if col.lower() in ['device_name', 'machine_name']:
                    machine_col = col
                    break
    
            if time_col and pulse_col:
                fig = px.line(
                    df, 
                    x=time_col, 
                    y=pulse_col, 
                    color=machine_col,
                    title="Pulse Per Minute Over Time",
                    labels={
                        time_col: "Time",
                        pulse_col: "Pulse Per Minute",
                        machine_col: "Machine"
                    }
                )
        
                # Enhanced formatting for pulse charts
                fig.update_layout(
                    height=600,
                    showlegend=True,
                    yaxis_title="Pulse Per Minute"
                )
        
                fig.update_xaxes(
                    tickangle=45,
                    tickformat='%H:%M'
                )
        
        # Display the chart
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            logger.info("Visualization created successfully")
            
            # Show data summary
            with st.expander("📊 Data Summary"):
                st.write(f"**Total records:** {len(df)}")
                if 'machine_name' in df.columns or 'device_name' in df.columns:
                    machine_col = 'machine_name' if 'machine_name' in df.columns else 'device_name'
                    unique_machines = df[machine_col].nunique()
                    st.write(f"**Unique machines:** {unique_machines}")
                    st.write(f"**Machines:** {', '.join(df[machine_col].unique())}")
                
                # Show efficiency statistics if it's an efficiency chart
                if chart_type == "efficiency_bar" and 'efficiency_percent' in df.columns:
                    st.write(f"**Average efficiency:** {df['efficiency_percent'].mean():.1f}%")
                    st.write(f"**Highest efficiency:** {df['efficiency_percent'].max():.1f}%")
                    st.write(f"**Lowest efficiency:** {df['efficiency_percent'].min():.1f}%")
                
                st.dataframe(df)
            
            return True
        else:
            error_msg = f"Could not create {chart_type} chart with available data."
            logger.error(error_msg)
            st.error(error_msg)
            st.write("**Available data:**")
            st.dataframe(df.head(10))
            return False
            
    except Exception as e:
        error_msg = f"Error creating visualization: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        st.write("**Debug information:**")
        st.write("Data sample:")
        st.dataframe(df.head() if not df.empty else pd.DataFrame())
        return False

def is_greeting_or_casual(user_query: str) -> bool:
    """Detect if the user query is a greeting or casual conversation"""
    user_query_lower = user_query.lower().strip()
    
    # Common greetings and casual phrases
    greetings = [
        'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
        'how are you', 'whats up', "what's up", 'yo', 'hiya', 'greetings'
    ]
    
    casual_phrases = [
        'thank you', 'thanks', 'bye', 'goodbye', 'see you', 'ok', 'okay',
        'cool', 'nice', 'great', 'awesome', 'perfect', 'got it', 'understand',
        'help', 'what can you do', 'how does this work', 'test'
    ]
    
    # Check if the query is just a greeting or casual phrase
    if user_query_lower in greetings + casual_phrases:
        return True
    
    # Check if query starts with greeting
    if any(user_query_lower.startswith(greeting) for greeting in greetings):
        return True
    
    # Check if it's a very short query without data-related keywords
    data_keywords = [
        'production', 'machine', 'data', 'show', 'chart', 'graph', 'plot',
        'select', 'table', 'database', 'query', 'april', 'month', 'day',
        'output', 'performance', 'efficiency', 'downtime', 'shift','pulse', 'pulse per minute', 'rate', 'length', 'variation', 'trend'
    ]
    
    if len(user_query_lower.split()) <= 3 and not any(keyword in user_query_lower for keyword in data_keywords):
        return True
    
    return False

def get_casual_response(user_query: str) -> str:
    """Generate appropriate responses for greetings and casual conversation"""
    user_query_lower = user_query.lower().strip()
    
    if any(greeting in user_query_lower for greeting in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']):
        return """Hello! 👋 I'm your Production Analytics Bot powered by Supabase. I'm here to help you analyze your production data and create visualizations."""

    elif any(phrase in user_query_lower for phrase in ['how are you', 'whats up', "what's up"]):
        return """I'm doing great, thank you! 😊 Ready to help you dive into your production data stored in Supabase.

Is there any specific production analysis or visualization you'd like me to help you with?"""

    elif any(phrase in user_query_lower for phrase in ['thank you', 'thanks']):
        return """You're welcome! 😊 I'm here whenever you need help with production data analysis or creating visualizations from your Supabase database.

Feel free to ask me about any production metrics you'd like to explore!"""

    elif any(phrase in user_query_lower for phrase in ['help', 'what can you do', 'how does this work']):
        return """I'm your Production Analytics Assistant powered by Supabase! Here's how I can help:

🔍 **Query your Supabase database** with natural language
📊 **Create interactive visualizations** (bar charts, line charts, pie charts)
🏭 **Analyze multi-machine production data** with comparisons
📈 **Track efficiency, output, and performance metrics**
⏱️ **Monitor pulse rates and time-series data**

Just ask me a question about your production data, and I'll generate both the analysis and visualizations for you!"""

    elif user_query_lower in ['test', 'testing']:
        return """System test successful! ✅ Supabase connection ready.

What would you like to analyze?"""

    else:
        return """I'm here to help with production data analysis and visualizations from your Supabase database! 

What production data would you like to explore? 📊"""

def get_enhanced_response(user_query: str, db: SQLDatabase, chat_history: list):
    """Enhanced response function with greeting detection and better error handling"""
    try:
        logger.info(f"Processing user query: {user_query}")
        
        # Step 0: Check if this is a greeting or casual conversation
        if is_greeting_or_casual(user_query):
            logger.info("Detected greeting/casual conversation")
            return get_casual_response(user_query)
        
        # Step 1: Detect visualization needs
        needs_viz, chart_type = detect_visualization_request(user_query)
        
        # Step 2: Generate SQL query with enhanced chain
        sql_chain = get_enhanced_sql_chain(db)
        sql_query = sql_chain.invoke({
            "question": user_query,
            "chat_history": chat_history
        })
        
        logger.info(f"Generated SQL query: {sql_query}")
        
        # Step 3: Execute SQL query with better error handling
        try:
            sql_response = db.run(sql_query)
            logger.info(f"SQL query executed successfully. Response length: {len(str(sql_response))}")
            
            # Check if response is empty
            if not sql_response or sql_response == "[]" or sql_response == "()":
                return "No data found for your query. This could be due to:\n\n1. **Date range issue**: The specified date range might not have data\n2. **Table structure**: Column names might be different\n3. **Data availability**: No records match your criteria\n\nPlease try:\n- Checking if data exists for the specified time period\n- Using a different date range\n- Asking about available tables or columns"
            
        except Exception as e:
            error_msg = f"SQL execution error: {str(e)}\n\n**Generated SQL Query:**\n```sql\n{sql_query}\n```\n\n**Possible issues:**\n1. Column names might be incorrect\n2. Table structure might be different\n3. Date format issues\n4. Data type mismatches"
            logger.error(error_msg)
            return error_msg
        
        # Step 4: Create visualization if needed
        chart_created = False
        if needs_viz:
            try:
                df = pd.read_sql(sql_query, db._engine)
                logger.info(f"DataFrame created with shape: {df.shape}")
                
                if df.empty:
                    st.warning("Query returned no data for visualization")
                else:
                    chart_created = create_enhanced_visualization(df, chart_type, user_query)
                    
            except Exception as e:
                error_msg = f"Visualization error: {str(e)}"
                logger.error(error_msg)
                st.error(error_msg)
        
        # Step 5: Generate natural language response
        template = """
        You are a data analyst providing insights about production data.
        
        Based on the SQL query results, provide a clear, informative response.
    
        SQL Query: {query}
        User Question: {question}
        SQL Response: {response}
        
        {visualization_note}
        
        Guidelines:
        1. Summarize key findings from the data
        2. Mention specific numbers/values when relevant
        3. If this is multi-machine data, highlight comparisons between machines
        4. If this is time-series data, mention trends or patterns
        5. Keep the response concise but informative
        
        
        # In the template section
        if 'pulse' in user_query.lower():
            visualization_note += " The pulse calculation shows the difference in length per minute, indicating machine activity rate."
        
        
        """
        
        visualization_note = ""
        if needs_viz and chart_created:
            visualization_note = "Note: The visualization above shows the data in an interactive chart format with different colors for each machine."
        elif needs_viz and not chart_created:
            visualization_note = "Note: I attempted to create a visualization but encountered formatting issues. The raw data is available above."
            
        
        prompt = ChatPromptTemplate.from_template(template)
    
        # Fixed Azure OpenAI configuration
        llm = AzureChatOpenAI(
            azure_endpoint=get_env_var("AZURE_OPENAI_ENDPOINT"),
            api_key=get_env_var("AZURE_OPENAI_API_KEY"),
            api_version=get_env_var("AZURE_OPENAI_API_VERSION"),
            azure_deployment=get_env_var("AZURE_OPENAI_DEPLOYMENT_NAME"),  # Changed from deployment_name
            temperature=0,  # Changed from 1 to 0 for more consistent SQL generation
            max_tokens=800,
    )
        
        
        chain = prompt | llm | StrOutputParser()
        
        response = chain.invoke({
            "question": user_query,
            "query": sql_query,
            "response": sql_response,
            "visualization_note": visualization_note
        })
        
        logger.info("Response generated successfully")
        return response
        
    except Exception as e:
        error_msg = f"An error occurred while processing your request: {str(e)}"
        logger.error(error_msg)
        return error_msg

# Streamlit UI
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm your Althinect Intelligence Bot. I can help you analyze multi-machine production data with colorful visualizations! 📊\n\nTry asking: *'Show all three machines production by each machine in April with bar chart for all 30 day'*"),
    ]

#load_dotenv()








if not get_env_var("AZURE_OPENAI_API_KEY"):
    st.error("⚠️ AZURE_OPENAI_API_KEY key not found. Please add AZURE_OPENAI_API_KEY to your .env file.")
    st.stop()

st.set_page_config(page_title="Althinect Intelligence Bot", page_icon="📊")

st.title("Althinect Intelligence Bot")

# Supabase sidebar
# Replace the entire sidebar section with this simplified version:

with st.sidebar:
    st.subheader("🔌 Database Connection")
    
    # Show connection status
    if "db" in st.session_state and st.session_state.db is not None:
        st.success("✅ Connected to Supabase!")
        
        # Optional: Show disconnect button
        if st.button("🔌 Reconnect Database", type="secondary"):
            try:
                with st.spinner("Reconnecting to database..."):
                    # Re-initialize database connection
                    db = init_supabase_database(
                        get_env_var("SUPABASE_URL"),
                        get_env_var("SUPABASE_ANON_KEY"),
                        get_env_var("SUPABASE_DB_PASSWORD")
                    )
                    st.session_state.db = db
                    st.success("✅ Database reconnected!")
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Reconnection failed: {str(e)}")
                st.session_state.db = None
    else:
        st.warning("🔴 Database Not Connected")
        
        if st.button("🔌 Connect to Database", type="primary"):
            try:
                with st.spinner("Connecting to database..."):
                    # Initialize Supabase client
                    supabase_client = init_supabase_client(
                        get_env_var("SUPABASE_URL"),
                        get_env_var("SUPABASE_ANON_KEY")
                    )
                    st.session_state.supabase_client = supabase_client
                    
                    # Initialize database connection
                    db = init_supabase_database(
                        get_env_var("SUPABASE_URL"),
                        get_env_var("SUPABASE_ANON_KEY"),
                        get_env_var("SUPABASE_DB_PASSWORD")
                    )
                    st.session_state.db = db
                    
                    st.success("✅ Connected to Supabase!")
                    logger.info("Supabase database connected successfully")
                    st.rerun()
                    
            except Exception as e:
                error_msg = f"❌ Database connection failed: {str(e)}"
                st.error(error_msg)
                logger.error(f"Database connection failed: {str(e)}")
                
                # Provide helpful error messages
                if "authentication failed" in str(e).lower():
                    st.info("💡 **Tip**: Check your database credentials in environment variables.")
                elif "could not translate host name" in str(e).lower():
                    st.info("💡 **Tip**: Verify your Supabase URL in environment variables.")
                elif "connection refused" in str(e).lower():
                    st.info("💡 **Tip**: Make sure your Supabase project is active.")

# Add this function right after the imports section to auto-initialize the database:

def auto_initialize_database():
    """Auto-initialize database connection if credentials are available"""
    try:
        # Check if all required environment variables are available
        supabase_url = get_env_var("SUPABASE_URL")
        supabase_key = get_env_var("SUPABASE_ANON_KEY") 
        db_password = get_env_var("SUPABASE_DB_PASSWORD")
        
        if supabase_url and supabase_key and db_password:
            if "db" not in st.session_state or st.session_state.db is None:
                logger.info("Auto-initializing database connection...")
                
                # Initialize Supabase client
                supabase_client = init_supabase_client(supabase_url, supabase_key)
                st.session_state.supabase_client = supabase_client
                
                # Initialize database connection
                db = init_supabase_database(supabase_url, supabase_key, db_password)
                st.session_state.db = db
                
                logger.info("Database auto-initialized successfully")
                return True
        else:
            logger.warning("Missing database credentials in environment variables")
            return False
            
    except Exception as e:
        logger.error(f"Auto-initialization failed: {str(e)}")
        return False

# Add this line right after st.set_page_config and before st.title:

# Auto-initialize database on app start
auto_initialize_database()

# Update the main title section to show connection status:

st.title("Althinect Intelligence Bot")

# Show connection status in main area
if "db" in st.session_state and st.session_state.db is not None:
    st.success("🟢 Database Connected - Ready to analyze your data!")
else:
    st.warning("🟡 Database not connected - Please connect using the sidebar")
        

# Chat interface
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            st.markdown(message.content)

user_query = st.chat_input("💬 Ask about multi-machine production data from Supabase...")
if user_query is not None and user_query.strip() != "":
    logger.info(f"User query received: {user_query}")
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_query)
        
    with st.chat_message("assistant", avatar="🤖"):
        # Check if it's a greeting first (no database needed)
        if is_greeting_or_casual(user_query):
            response = get_casual_response(user_query)
            st.markdown(response)
        elif "db" in st.session_state:
            with st.spinner("🔄 Analyzing Supabase data and creating visualization..."):
                response = get_enhanced_response(user_query, st.session_state.db, st.session_state.chat_history)
                st.markdown(response)
        else:
            response = "⚠️ Please connect to your Supabase database first using the sidebar to analyze production data."
            st.markdown(response)
            logger.warning("User attempted to query without Supabase connection")
        
    st.session_state.chat_history.append(AIMessage(content=response))
    logger.info("Conversation turn completed")