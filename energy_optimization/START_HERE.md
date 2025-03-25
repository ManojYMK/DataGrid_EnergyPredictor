# Alberta Energy Optimization System - Quick Start

Welcome to the Energy Consumption Optimization System! This tool analyzes Alberta's energy grid data to identify opportunities for increasing electricity production efficiency.

## What's Included

This system contains:

1. **Data Extraction**: Automatically extracts data from Alberta's energy grid report (CSDReportServlet.pdf)
2. **Machine Learning Models**: Predicts optimal efficiency for each energy asset
3. **Optimization Recommendations**: Identifies which assets have the most potential for improvement
4. **Visualizations**: Shows efficiency gaps and potential improvements graphically

## Quick Start Guide

### Step 1: Install Dependencies (First-time Only)

Before running the system, make sure the required dependencies are installed:

```bash
pip3 install PyPDF2 pandas numpy scikit-learn torch matplotlib
```

Or simply run the optimization script, which will install dependencies automatically.

### Step 2: Run the System

1. Open Terminal
2. Navigate to this directory:
   ```bash
   cd /Users/manoj/Desktop/AI\ Editor/energy_optimization/
   ```
3. Make the script executable (first-time only):
   ```bash
   chmod +x run_optimization.sh
   ```
4. Run the script:
   ```bash
   ./run_optimization.sh
   ```

The system will:
- Check for and install required packages
- Process the PDF data
- Build machine learning models
- Generate recommendations
- Create visualizations
- Output a detailed report

### Step 3: View the Results

1. **Main Report**: Open `reports/optimization_report.txt` to see the top recommendations
2. **Visualizations**: Check the `plots/` directory for graphs and charts
3. **Extracted Data**: Browse the `data/` directory to see the raw data

## Troubleshooting

If you encounter an error about missing dependencies (like "No module named 'PyPDF2'"), run:

```bash
pip3 install PyPDF2 pandas numpy scikit-learn torch matplotlib
```

For more detailed troubleshooting, see `INSTALLATION.md`.

## Learn More

- **README.md**: Technical details about how the system works
- **ENERGY_OPTIMIZATION_GUIDE.md**: Non-technical explanation for stakeholders
- **INSTALLATION.md**: Detailed setup instructions

## Key Files

- `main.py`: Main entry point for the system
- `pdf_data_extractor.py`: Extracts data from the PDF report
- `energy_optimizer.py`: Contains machine learning models and optimization logic
- `run_optimization.sh`: Shell script to simplify running the system

## Requirements

- Python 3.6+
- PDF file at: `/Users/manoj/Desktop/AI Editor/CSDReportServlet.pdf`
- Required Python packages (automatically installed by the script)

## For Technical Users

If you want to run specific parts of the pipeline:

```python
# Extract data only
python3 -c "from pdf_data_extractor import EnergyDataExtractor; e = EnergyDataExtractor('../CSDReportServlet.pdf'); e.extract_data(); e.save_to_csv('data')"

# Run the complete pipeline
python3 main.py
```

## Contact

For questions or issues, please refer to the troubleshooting section in `INSTALLATION.md`. 