#!/usr/bin/env python3
"""
PDF Data Extractor for Energy Optimization

This script extracts energy generation and consumption data from the CSDReportServlet.pdf file,
which contains Alberta's Current Supply Demand Report.
"""

import os
import re
import pandas as pd
import PyPDF2
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("extraction.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class EnergyDataExtractor:
    """Class to extract energy data from PDF reports."""
    
    def __init__(self, pdf_path):
        """Initialize with the path to the PDF file."""
        self.pdf_path = pdf_path
        self.summary_data = {}
        self.generation_by_group = pd.DataFrame()
        self.interchange_data = pd.DataFrame()
        self.asset_data = {}
        
    def extract_data(self):
        """Extract all data from the PDF file."""
        logger.info(f"Extracting data from {self.pdf_path}...")
        
        try:
            # Open the PDF file
            pdf_reader = PyPDF2.PdfReader(self.pdf_path)
            
            # Get the total number of pages
            num_pages = len(pdf_reader.pages)
            logger.info(f"PDF has {num_pages} pages.")
            
            # Extract text from all pages
            all_text = ""
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                all_text += page.extract_text() + "\n"
            
            # Extract different sections of data
            self._extract_summary_data(all_text)
            self._extract_generation_by_group(all_text)
            self._extract_interchange_data(all_text)
            self._extract_asset_data(all_text)
            
            logger.info("Data extraction completed successfully.")
            return True
            
        except Exception as e:
            logger.error(f"Error extracting data: {str(e)}")
            return False
    
    def _extract_summary_data(self, text):
        """Extract summary data from the PDF."""
        try:
            # Look for the SUMMARY section
            summary_pattern = r"SUMMARY(.*?)GENERATION"
            summary_match = re.search(summary_pattern, text, re.DOTALL)
            
            if summary_match:
                summary_text = summary_match.group(1)
                
                # Extract key-value pairs
                patterns = {
                    'total_net_generation': r"Alberta Total Net Generation\s*(\d+)",
                    'net_actual_interchange': r"Net Actual Interchange\s*(\d+)",
                    'alberta_internal_load': r"Alberta Internal Load \(AIL\)\s*(\d+)",
                    'net_to_grid_generation': r"Net-To-Grid Generation\s*(\d+)",
                    'contingency_reserve_required': r"Contingency Reserve Required\s*(\d+)",
                    'dispatched_contingency_reserve': r"Dispatched Contingency Reserve\s*\(DCR\)\s*(\d+)",
                    'dispatched_contingency_reserve_gen': r"Dispatched Contingency Reserve -\s*Gen\s*(\d+)",
                    'dispatched_contingency_reserve_other': r"Dispatched Contingency Reserve -\s*Other\s*(\d+)",
                    'ffr_armed_dispatch': r"FFR Armed Dispatch\s*(\d+)",
                    'ffr_offered_volume': r"FFR Offered Volume\s*(\d+)",
                    'long_lead_time_volume': r"Long Lead Time Volume\s*(\d+)"
                }
                
                for key, pattern in patterns.items():
                    match = re.search(pattern, summary_text)
                    if match:
                        self.summary_data[key] = int(match.group(1))
                    else:
                        self.summary_data[key] = None
                        
                logger.info(f"Extracted summary data: {self.summary_data}")
            else:
                logger.warning("SUMMARY section not found in the PDF.")
                
        except Exception as e:
            logger.error(f"Error extracting summary data: {str(e)}")
    
    def _extract_generation_by_group(self, text):
        """Extract generation data by group."""
        try:
            # Look for the GENERATION GROUP section
            generation_pattern = r"GENERATION\s*GROUP\s*MC\s*TNG\s*DCR(.*?)INTERCHANGE"
            generation_match = re.search(generation_pattern, text, re.DOTALL)
            
            if generation_match:
                generation_text = generation_match.group(1)
                
                # Split into lines and process
                lines = generation_text.strip().split('\n')
                data = []
                
                for line in lines:
                    # Match the group name and values
                    match = re.match(r"(\w+(?:\s+\w+)*)\s+(\d+)\s+(\d+)\s+(\d+)", line)
                    if match:
                        group, mc, tng, dcr = match.groups()
                        data.append({
                            'group': group,
                            'maximum_capability': int(mc),
                            'total_net_generation': int(tng),
                            'dispatched_contingency_reserve': int(dcr)
                        })
                
                self.generation_by_group = pd.DataFrame(data)
                logger.info(f"Extracted generation data by group: {len(self.generation_by_group)} rows")
            else:
                logger.warning("GENERATION GROUP section not found in the PDF.")
                
        except Exception as e:
            logger.error(f"Error extracting generation by group data: {str(e)}")
    
    def _extract_interchange_data(self, text):
        """Extract interchange data."""
        try:
            # Look for the INTERCHANGE section
            interchange_pattern = r"INTERCHANGE\s*PATH\s*ACTUAL\s*FLOW(.*?)(?:GAS|SOLAR|WIND|HYDRO)"
            interchange_match = re.search(interchange_pattern, text, re.DOTALL)
            
            if interchange_match:
                interchange_text = interchange_match.group(1)
                
                # Split into lines and process
                lines = interchange_text.strip().split('\n')
                data = []
                
                for line in lines:
                    # Match the path and flow
                    match = re.match(r"(\w+(?:\s+\w+)*)\s+(-?\d+)", line)
                    if match:
                        path, flow = match.groups()
                        data.append({
                            'path': path,
                            'actual_flow': int(flow)
                        })
                
                self.interchange_data = pd.DataFrame(data)
                logger.info(f"Extracted interchange data: {len(self.interchange_data)} rows")
            else:
                logger.warning("INTERCHANGE section not found in the PDF.")
                
        except Exception as e:
            logger.error(f"Error extracting interchange data: {str(e)}")
    
    def _extract_asset_data(self, text):
        """Extract asset-specific data."""
        try:
            # Define the asset categories to extract
            categories = ['GAS', 'HYDRO', 'ENERGY STORAGE', 'SOLAR', 'WIND', 'COGENERATION', 'COMBINED CYCLE', 'BIOMASS AND OTHER']
            
            for category in categories:
                assets = []
                
                # Look for the category section
                pattern = f"{category}.*?ASSET\\s+MC\\s*TNG\\s*DCR(.*?)(?:{'|'.join(categories)}|Report Disclaimer)"
                category_match = re.search(pattern, text, re.DOTALL)
                
                if category_match:
                    category_text = category_match.group(1)
                    
                    # Extract asset data using regex
                    asset_pattern = r"([^0-9\n]+)(?:\([A-Z0-9]+\))(?:\^|\*)?\s+(\d+)\s+(\d+)\s+(\d+)"
                    asset_matches = re.finditer(asset_pattern, category_text)
                    
                    for match in asset_matches:
                        name, mc, tng, dcr = match.groups()
                        assets.append({
                            'name': name.strip(),
                            'maximum_capability': int(mc),
                            'total_net_generation': int(tng),
                            'dispatched_contingency_reserve': int(dcr)
                        })
                    
                    self.asset_data[category] = pd.DataFrame(assets)
                    logger.info(f"Extracted {len(assets)} {category} assets")
                else:
                    logger.warning(f"{category} section not found in the PDF.")
                    self.asset_data[category] = pd.DataFrame()
                    
        except Exception as e:
            logger.error(f"Error extracting asset data: {str(e)}")

    def save_to_csv(self, output_dir="data"):
        """Save extracted data to CSV files."""
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save summary data
            summary_df = pd.DataFrame([self.summary_data])
            summary_df.to_csv(f"{output_dir}/summary.csv", index=False)
            
            # Save generation by group data
            self.generation_by_group.to_csv(f"{output_dir}/generation_by_group.csv", index=False)
            
            # Save interchange data
            self.interchange_data.to_csv(f"{output_dir}/interchange.csv", index=False)
            
            # Save asset data for each category
            for category, df in self.asset_data.items():
                if not df.empty:
                    # Clean category name for filename
                    category_name = category.lower().replace(' ', '_')
                    df.to_csv(f"{output_dir}/{category_name}_assets.csv", index=False)
            
            # Create a combined assets dataframe
            all_assets = []
            for category, df in self.asset_data.items():
                if not df.empty:
                    df['category'] = category
                    all_assets.append(df)
            
            if all_assets:
                combined_assets = pd.concat(all_assets, ignore_index=True)
                combined_assets.to_csv(f"{output_dir}/all_assets.csv", index=False)
                logger.info(f"Saved {len(combined_assets)} assets to CSV")
            
            logger.info(f"All data saved to directory: {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving data to CSV: {str(e)}")
            return False

if __name__ == "__main__":
    # PDF path
    pdf_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CSDReportServlet.pdf")
    
    # Create extractor and process data
    extractor = EnergyDataExtractor(pdf_path)
    if extractor.extract_data():
        # Save extracted data to CSV files
        output_dir = os.path.join(os.path.dirname(__file__), "data")
        extractor.save_to_csv(output_dir) 