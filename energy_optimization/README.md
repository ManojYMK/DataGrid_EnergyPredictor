# Energy Consumption Optimization System

This project implements a machine learning and deep learning-based system for optimizing energy consumption based on the Alberta energy grid data provided in the CSDReportServlet.pdf report. The system analyzes the data, identifies optimization opportunities, and provides actionable recommendations to improve energy efficiency.

## Project Structure

```
energy_optimization/
├── pdf_data_extractor.py    # Extracts data from the PDF report
├── energy_optimizer.py      # ML/DL models for energy optimization
├── main.py                  # Main script to run the entire pipeline
├── data/                    # Extracted data in CSV format
├── plots/                   # Visualization outputs
├── reports/                 # Generated recommendation reports
└── README.md                # This file
```

## How It Works

The system operates in three main stages:

1. **Data Extraction**: Parses the CSDReportServlet.pdf file to extract:
   - Summary statistics of Alberta's energy grid
   - Generation data by energy source group
   - Interchange data between provinces/states
   - Detailed asset-specific data across multiple energy categories

2. **Data Analysis and Modeling**: Applies various machine learning techniques to:
   - Calculate and predict efficiency ratios for each energy asset
   - Identify optimization potential based on maximum capacity vs. current generation
   - Build predictive models using:
     - Linear Regression
     - Random Forest
     - Gradient Boosting
     - Neural Networks (deep learning with PyTorch)

3. **Recommendation Generation**: Creates:
   - Prioritized list of assets with the highest optimization potential
   - Visualizations of efficiency gaps and improvement opportunities
   - Comprehensive report with actionable recommendations

## Quick Start

To run the full optimization pipeline:

```bash
cd /Users/manoj/Desktop/AI\ Editor/energy_optimization/
python3 main.py
```

This will:
1. Extract data from the PDF report
2. Train machine learning and deep learning models
3. Generate optimization recommendations
4. Create visualizations in the `plots` directory
5. Output a detailed report in the `reports` directory

## Key Features

- **Automated Data Extraction**: Extracts complex tabular data from PDF using regular expressions
- **Multiple ML Models**: Compares various algorithms to find the best performing model
- **Deep Learning Integration**: Uses PyTorch to build neural networks for enhanced prediction accuracy
- **Efficiency Calculation**: Determines current vs. potential efficiency for each energy asset
- **Visual Reports**: Generates charts and graphs to visualize optimization opportunities
- **Prioritized Recommendations**: Ranks assets by potential impact for focused optimization efforts
- **Demo Mode**: If asset data cannot be extracted from the PDF, the system automatically generates sample data for demonstration purposes

## Technical Details

### Input Variables

The system uses the following input variables from each energy asset:

- **Maximum Capability (MC)**: The maximum power an asset can generate (in MW)
- **Total Net Generation (TNG)**: The actual power currently being generated (in MW)
- **Dispatched Contingency Reserve (DCR)**: Reserved capacity for grid stability (in MW)
- **Energy Category**: The type of energy source (Wind, Solar, Hydro, etc.)

### Output Variables

The system predicts and analyzes:

- **Efficiency Ratio**: Calculated as TNG/MC, representing how efficiently each asset uses its capacity
- **Optimization Potential**: The gap between maximum capability and current generation
- **Potential Generation Increase**: The additional energy that could be generated with improved efficiency

### Model Evaluation

The system evaluates models using:

- **Mean Squared Error (MSE)**: To measure prediction accuracy
- **R² Score**: To measure how well the model explains the variance in the data

## Visualizations

The system generates several visualizations:

1. **Model Comparison Charts**: Compare the performance of different ML/DL algorithms
2. **Top Assets by Potential**: Bar chart of assets with highest optimization potential
3. **Efficiency Comparison**: Current vs. potential efficiency for top assets
4. **Category Potential**: Aggregated potential by energy source category
5. **Neural Network Learning Curves**: Training progress visualization for deep learning models

## Optimization Report

The generated report includes:

- Summary statistics of current and potential generation
- Detailed recommendations for the top 10 assets with highest potential
- Current and predicted efficiency ratios
- Potential generation increase in MW
- Overall potential percentage increase in generation

## Sample Data Mode

If the system cannot extract asset-specific data from the PDF (which might happen with certain PDF formats or if the PDF doesn't contain the expected data), it will automatically generate sample data based on:

1. The extracted generation group data (if available)
2. Completely synthetic data based on typical energy generation patterns if no group data is available

This allows the system to still demonstrate its optimization capabilities even when working with limited data. A notification will be displayed when sample data is being used.

## Dependencies

- Python 3.6 or higher
- PyPDF2: For PDF text extraction
- pandas: For data manipulation
- numpy: For numerical operations
- scikit-learn: For machine learning models
- PyTorch: For deep learning models
- matplotlib: For visualization

## Limitations

- The analysis is based on a single snapshot of data from the PDF report
- More accurate models could be built with historical time-series data
- Real-world implementation would require additional operational constraints 