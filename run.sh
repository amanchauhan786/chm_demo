#!/bin/bash
# Run the CHM Dashboard

echo "🌳 Starting CHM Analysis Dashboard..."
echo "====================================="

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Streamlit not found. Installing dependencies..."
    pip install -r requirements.txt
fi

# Run the application
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
