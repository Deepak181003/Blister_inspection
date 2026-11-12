"""
report.py
=========
Module responsible for generating structured CSV inspection reports,
logging defect details, and computing quality control batch statistics.

Columns tracked:
  - Image Filename
  - Inspection Timestamp
  - Defect Category
  - Defect Count
  - Inspection Decision (PASS / REJECT / REVIEW)
  - Processing Time (ms)
  - Defect Descriptions / Locations
"""

import os
import csv
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
from detection import InspectionResult


DEFAULT_REPORT_PATH = os.path.join("results", "inspection_report.csv")


def init_report_file(csv_path: str = DEFAULT_REPORT_PATH) -> None:
    """
    Initializes the inspection CSV report file with standard column headers
    if the file does not already exist.

    Args:
        csv_path (str): Destination file path for the CSV report.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    headers = [
        "Image_Filename",
        "Inspection_Timestamp",
        "Defect_Category",
        "Defect_Count",
        "Inspection_Decision",
        "Processing_Time_ms",
        "Defect_Details"
    ]
    
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)


def log_inspection_result(result: InspectionResult,
                          csv_path: str = DEFAULT_REPORT_PATH) -> Dict[str, Any]:
    """
    Appends a single image inspection result to the master CSV report.

    Args:
        result (InspectionResult): The inspection output object from detection.py.
        csv_path (str): Filepath where the CSV record should be logged.

    Returns:
        Dict[str, Any]: Dictionary representing the row appended to the CSV.
    """
    init_report_file(csv_path)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    base_filename = os.path.basename(result.image_filename)

    # Determine defect categories present
    if result.total_defects == 0:
        category_str = "NONE"
        details_str = "All cavities and blister seals passed quality check."
    else:
        unique_categories = sorted(list(set(d.defect_type for d in result.defects)))
        category_str = "; ".join(unique_categories)
        details_str = " | ".join(d.description for d in result.defects)

    row_data = {
        "Image_Filename": base_filename,
        "Inspection_Timestamp": timestamp,
        "Defect_Category": category_str,
        "Defect_Count": result.total_defects,
        "Inspection_Decision": result.overall_decision,
        "Processing_Time_ms": result.processing_time_ms,
        "Defect_Details": details_str
    }

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            row_data["Image_Filename"],
            row_data["Inspection_Timestamp"],
            row_data["Defect_Category"],
            row_data["Defect_Count"],
            row_data["Inspection_Decision"],
            row_data["Processing_Time_ms"],
            row_data["Defect_Details"]
        ])

    return row_data


def generate_batch_summary(csv_path: str = DEFAULT_REPORT_PATH) -> Optional[pd.DataFrame]:
    """
    Reads the inspection CSV log using Pandas and returns a formatted DataFrame.

    Why Pandas is used:
      - Clean tabular presentation for terminal display and analysis.
      - Easy aggregation of inspection metrics (Total Inspected, Pass Rate, Defect Breakdown).

    Args:
        csv_path (str): Path to inspection CSV file.

    Returns:
        Optional[pd.DataFrame]: DataFrame of all logged records, or None if empty.
    """
    if not os.path.exists(csv_path):
        print(f"[REPORT] No report file found at {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    return df


def print_batch_metrics(csv_path: str = DEFAULT_REPORT_PATH) -> None:
    """
    Prints a clean, formatted executive summary of inspection results.

    Metrics displayed:
      - Total Blister Packs Inspected
      - Total Passed, Rejected, and Under Review
      - Pass Rate (%)
      - Average Processing Latency (ms)
      - Defect Type Breakdown
    """
    df = generate_batch_summary(csv_path)
    if df is None or df.empty:
        print("[REPORT] No inspection records available to summarize.")
        return

    total = len(df)
    decision_counts = df["Inspection_Decision"].value_counts().to_dict()
    passed = decision_counts.get("PASS", 0)
    rejected = decision_counts.get("REJECT", 0)
    review = decision_counts.get("REVIEW", 0)
    pass_rate = (passed / total) * 100.0 if total > 0 else 0.0
    avg_latency = df["Processing_Time_ms"].mean()

    print("\n" + "=" * 65)
    print("        PHARMACEUTICAL BLISTER PACK INSPECTION SUMMARY")
    print("=" * 65)
    print(f" Total Packs Inspected  : {total}")
    print(f" PASS Decisions         : {passed} ({pass_rate:.1f}%)")
    print(f" REJECT Decisions       : {rejected} ({(rejected/total)*100:.1f}%)")
    print(f" REVIEW Decisions       : {review} ({(review/total)*100:.1f}%)")
    print(f" Average Latency        : {avg_latency:.2f} ms per pack")
    print("-" * 65)
    print(" Defect Categories Breakdown:")
    
    # Extract individual defect frequencies
    cat_series = df[df["Defect_Category"] != "NONE"]["Defect_Category"]
    if not cat_series.empty:
        split_cats = []
        for item in cat_series:
            split_cats.extend([c.strip() for c in item.split(";")])
        cat_counts = pd.Series(split_cats).value_counts()
        for cat, cnt in cat_counts.items():
            print(f"   * {cat:<24} : {cnt} occurrence(s)")
    else:
        print("   * No defects recorded across batch.")
        
    print("=" * 65 + "\n")
