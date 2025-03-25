#!/usr/bin/env python3
"""
Energy Optimization Main Script

This script runs the complete energy optimization pipeline:
1. Extracts data from the Alberta energy grid PDF report
2. Applies machine learning and deep learning models to optimize energy consumption
3. Generates visualizations and reports
"""

import os
import sys
import logging

# Check for required packages before proceeding
try:
    # Check for PyPDF2 first since that's the one causing issues
    import PyPDF2
except ImportError:
    print("\nERROR: Required package 'PyPDF2' is not installed.")
    print("Please install it with one of the following commands:")
    print("    pip3 install PyPDF2")
    print("    python3 -m pip install PyPDF2")
    print("\nOr run the script './run_optimization.sh' which will install all dependencies.\n")
    sys.exit(1)

# Check for other required packages
required_packages = ['pandas', 'numpy', 'sklearn', 'torch', 'matplotlib']
missing_packages = []

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print("\nERROR: The following required packages are not installed:")
    for package in missing_packages:
        print(f"  - {package}")
    print("\nPlease install them with:")
    print(f"    pip3 install {' '.join(missing_packages)}")
    print("\nOr run the script './run_optimization.sh' which will install all dependencies.\n")
    sys.exit(1)

# Now proceed with the imports for the main functionality
from pdf_data_extractor import EnergyDataExtractor
from energy_optimizer import run_optimization_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("energy_optimization.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Run the full pipeline for energy optimization."""
    try:
        # Current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # PDF path (relative to the main project directory)
        pdf_path = os.path.join(os.path.dirname(current_dir), "CSDReportServlet.pdf")
        
        # Check if PDF exists
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return False
        
        # Data directory
        data_dir = os.path.join(current_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Step 1: Extract data from PDF
        logger.info("Step 1: Extracting data from PDF...")
        extractor = EnergyDataExtractor(pdf_path)
        if not extractor.extract_data():
            logger.error("Failed to extract data from PDF")
            return False
        
        # Save extracted data to CSV files
        if not extractor.save_to_csv(data_dir):
            logger.error("Failed to save extracted data to CSV")
            return False
        
        logger.info(f"Data extraction complete. Data saved to {data_dir}")
        
        # Check if asset data was successfully extracted
        if all(df.empty for df in extractor.asset_data.values()):
            print("\n" + "=" * 80)
            print("NOTE: No actual asset data was found in the PDF.")
            print("Sample asset data will be generated for demonstration purposes.")
            print("=" * 80 + "\n")
            logger.warning("No actual asset data found in PDF. Using sample data for demonstration.")
        
        # Step 2: Run optimization pipeline
        logger.info("Step 2: Running energy optimization pipeline...")
        if not run_optimization_pipeline():
            logger.error("Energy optimization pipeline failed")
            return False
        
        logger.info("Energy optimization pipeline completed successfully")
        
        # Print summary of results
        reports_dir = os.path.join(current_dir, "reports")
        report_path = os.path.join(reports_dir, "optimization_report.txt")
        
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                print("\n" + "=" * 80)
                print("ENERGY OPTIMIZATION RESULTS SUMMARY")
                print("=" * 80)
                print(f.read())
        
        # List visualization files
        plots_dir = os.path.join(current_dir, "plots")
        if os.path.exists(plots_dir):
            print("\nVisualization files generated:")
            for filename in os.listdir(plots_dir):
                if filename.endswith(".png"):
                    print(f"- {os.path.join(plots_dir, filename)}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 