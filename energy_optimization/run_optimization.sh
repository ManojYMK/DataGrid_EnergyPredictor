#!/bin/bash

# Energy Optimization System Runner
# This script simplifies running the energy optimization system

echo "=================================================="
echo "    Alberta Energy Consumption Optimization"
echo "=================================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in the PATH."
    echo "Please install Python 3 to use this system."
    exit 1
fi

# Check if the PDF file exists
PDF_PATH="../CSDReportServlet.pdf"
if [ ! -f "$PDF_PATH" ]; then
    echo "Error: Required PDF file not found: $PDF_PATH"
    echo "Please ensure the CSDReportServlet.pdf file is in the correct location."
    exit 1
fi

# Create required directories
mkdir -p data plots reports

# Install required packages if needed
echo "Installing required packages..."
echo "This may take a moment..."

# Try to install packages with pip3 first (more reliable on macOS)
if command -v pip3 &> /dev/null; then
    pip3 install PyPDF2 pandas numpy scikit-learn torch matplotlib
    INSTALL_STATUS=$?
# Fall back to python3 -m pip if pip3 is not available
else
    python3 -m pip install PyPDF2 pandas numpy scikit-learn torch matplotlib
    INSTALL_STATUS=$?
fi

# Check if installation was successful
if [ $INSTALL_STATUS -ne 0 ]; then
    echo ""
    echo "Error: Failed to install required packages."
    echo "Please try installing them manually with:"
    echo "pip3 install PyPDF2 pandas numpy scikit-learn torch matplotlib"
    exit 1
else
    echo "Required packages installed successfully."
fi

# Run the optimization pipeline
echo ""
echo "Starting energy optimization pipeline..."
echo "This may take a few minutes to complete."
echo ""

python3 main.py

# Check if the run was successful
if [ $? -ne 0 ]; then
    echo ""
    echo "Error: The energy optimization pipeline encountered an error."
    echo "Please check the logs for details."
    exit 1
fi

echo ""
echo "=================================================="
echo "    Optimization Complete!"
echo "=================================================="
echo ""
echo "Results are available in:"
echo "- Reports: ./reports/optimization_report.txt"
echo "- Visualizations: ./plots/ directory"
echo ""
echo "To view visualizations, open the PNG files in the plots directory."
echo "To see the full technical details, review the README.md file."
echo ""
echo "Thank you for using the Alberta Energy Optimization System!" 