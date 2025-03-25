# Installation Guide for Energy Optimization System

This guide will help you set up and run the Energy Consumption Optimization System successfully.

## Prerequisites

The system requires:

- Python 3.6 or higher
- Pip (Python package manager)
- The CSDReportServlet.pdf file in the parent directory

## Step 1: Install Python Dependencies

You can install all required dependencies by running:

```bash
# For most systems
pip3 install PyPDF2 pandas numpy scikit-learn torch matplotlib

# If the above doesn't work, try:
python3 -m pip install PyPDF2 pandas numpy scikit-learn torch matplotlib
```

Or, use the automated script which will install dependencies for you:

```bash
./run_optimization.sh
```

### macOS Specific Notes

On macOS, you might need to use `pip3` explicitly instead of `pip`:

```bash
pip3 install PyPDF2 pandas numpy scikit-learn torch matplotlib
```

If you encounter permission issues, you can add the `--user` flag:

```bash
pip3 install --user PyPDF2 pandas numpy scikit-learn torch matplotlib
```

## Step 2: Ensure PDF File Is Available

The system analyzes data from the CSDReportServlet.pdf file. This file should be located at:

```
/Users/manoj/Desktop/AI Editor/CSDReportServlet.pdf
```

If the file is in a different location, you can either:
1. Move it to the expected location
2. Update the file path in `main.py`

## Step 3: Run the System

### Option 1: Using the Script (Recommended)

The easiest way to run the system is with the provided shell script:

```bash
cd /Users/manoj/Desktop/AI\ Editor/energy_optimization/
chmod +x run_optimization.sh  # Make executable (first time only)
./run_optimization.sh
```

### Option 2: Direct Python Execution

You can also run the Python script directly:

```bash
cd /Users/manoj/Desktop/AI\ Editor/energy_optimization/
python3 main.py
```

## Step 4: Review the Results

After successful execution, the system will generate:

1. Extracted data files in the `data/` directory
2. Visualization plots in the `plots/` directory
3. A detailed report in the `reports/` directory

The main report file will be at:
```
/Users/manoj/Desktop/AI Editor/energy_optimization/reports/optimization_report.txt
```

## Troubleshooting

If you encounter issues:

### ModuleNotFoundError: No module named 'PyPDF2'

This is a common error when the PyPDF2 package isn't properly installed. Try these solutions:

1. Install explicitly with pip3:
   ```bash
   pip3 install PyPDF2
   ```

2. If you're using a virtual environment, make sure it's activated.

3. Check your Python version and ensure you're using the same version to install packages and run the script:
   ```bash
   python3 --version
   pip3 --version
   ```

4. On some systems, you may need to use pip3 with the Python module flag:
   ```bash
   python3 -m pip install PyPDF2
   ```

### Other Missing Dependencies

If you see errors about other missing Python modules, install them individually:

```bash
pip3 install <module_name>
```

### PDF File Not Found

If the system can't locate the PDF file:
1. Confirm the file exists and is correctly named
2. Check that it's in the parent directory of the energy_optimization folder
3. If needed, update the path in `main.py`

### Permission Errors

If you get permission errors when running the script:
1. Ensure the script is executable: `chmod +x run_optimization.sh`
2. Run with sudo if necessary: `sudo ./run_optimization.sh`

## Getting Help

If you need assistance:
1. Check the log files (*.log) for detailed error messages
2. Refer to the README.md for system overview
3. Review the ENERGY_OPTIMIZATION_GUIDE.md for a non-technical explanation 