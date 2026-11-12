"""
main.py
=======
Main Application Entry Point for Automated Pharmaceutical Blister Pack Defect Detection.

Features:
  1. Interactive Console Menu + File Selection Dialog
  2. Single Image Inspection & Multi-Panel Visualization
  3. Batch Inspection Across Dataset Directories
  4. Real-time Defect Localization & Classification (PASS / REJECT / REVIEW)
  5. Automatic CSV Inspection Report Generation
  6. Synthetic Test Dataset Generator Integration

Visualization Pipeline (5-Panel Display):
  [1] Original Image (BGR -> RGB)
  [2] Grayscale Image
  [3] Threshold Image (Binarized)
  [4] Canny Edge Detection
  [5] Final Defect Detection (With localized bounding boxes & decision banner)
"""

import os
import sys
import argparse
import glob
import subprocess
from typing import Optional
import cv2
import matplotlib.pyplot as plt

from preprocessing import load_image, preprocess_pipeline
from detection import inspect_blister_pack, InspectionResult, BlisterPackReference
from report import log_inspection_result, print_batch_metrics, DEFAULT_REPORT_PATH
from generate_dataset import build_full_dataset


def display_five_panel_visualization(image_path: str,
                                     result: InspectionResult,
                                     save_to_results: bool = True,
                                     show_plot: bool = True) -> str:
    """
    Renders and displays the required 5-panel computer vision inspection dashboard:
      Panel 1: Original Image
      Panel 2: Grayscale Image
      Panel 3: Threshold (Binarized) Image
      Panel 4: Canny Edge Detection Image
      Panel 5: Final Defect Detection Image (Annotated)

    Args:
        image_path (str): Path to original blister pack image.
        result (InspectionResult): Detection output object.
        save_to_results (bool): Whether to save figure to 'results/' folder.
        show_plot (bool): Whether to trigger interactive plt.show() window.

    Returns:
        str: File path to saved visualization image.
    """
    # Run preprocessing pipeline to obtain all intermediate stages
    raw_bgr = load_image(image_path)
    stages = preprocess_pipeline(raw_bgr)

    # Convert BGR images to RGB for Matplotlib rendering
    rgb_original = cv2.cvtColor(stages["original"], cv2.COLOR_BGR2RGB)
    rgb_annotated = cv2.cvtColor(result.annotated_image, cv2.COLOR_BGR2RGB)

    # Create figure with 2 rows x 3 columns layout (top row: 3 panels, bottom row: 2 larger panels)
    fig = plt.figure(figsize=(15, 9))
    fig.canvas.manager.set_window_title(f"Blister Pack Inspection - {os.path.basename(image_path)}")

    # 1. Original Image
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(rgb_original)
    ax1.set_title("1. Original Image", fontsize=11, fontweight="bold")
    ax1.axis("off")

    # 2. Grayscale Conversion
    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(stages["gray"], cmap="gray")
    ax2.set_title("2. Grayscale Image", fontsize=11, fontweight="bold")
    ax2.axis("off")

    # 3. Adaptive Thresholding
    ax3 = plt.subplot(2, 3, 3)
    ax3.imshow(stages["thresh"], cmap="gray")
    ax3.set_title("3. Threshold (Binarized) Image", fontsize=11, fontweight="bold")
    ax3.axis("off")

    # 4. Canny Edge Detection
    ax4 = plt.subplot(2, 3, 4)
    ax4.imshow(stages["edges"], cmap="gray")
    ax4.set_title("4. Canny Edge Detection", fontsize=11, fontweight="bold")
    ax4.axis("off")

    # 5. Final Defect Detection Image (Larger span on bottom right)
    ax5 = plt.subplot(2, 3, (5, 6))
    ax5.imshow(rgb_annotated)
    decision_color = "green" if result.overall_decision == "PASS" else ("red" if result.overall_decision == "REJECT" else "orange")
    ax5.set_title(
        f"5. Final Defect Detection [{result.overall_decision}] - {result.total_defects} Defect(s) ({result.processing_time_ms} ms)",
        fontsize=12,
        fontweight="bold",
        color=decision_color
    )
    ax5.axis("off")

    plt.tight_layout()

    # Save visualization to results directory
    saved_path = ""
    if save_to_results:
        os.makedirs("results", exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        saved_path = os.path.join("results", f"inspection_{base_name}.png")
        plt.savefig(saved_path, dpi=200, bbox_inches="tight")
        print(f"[VISUALIZATION] Dashboard saved to: '{saved_path}'")

    if show_plot:
        try:
            plt.show()
        except Exception as e:
            print(f"[VISUALIZATION] Interactive plot window skipped: {e}")

    plt.close(fig)
    return saved_path


def run_single_inspection(image_path: str,
                          reference_image_path: Optional[str] = None,
                          show_ui: bool = True) -> InspectionResult:
    """
    Executes inspection on a single blister image, prints diagnostics, logs to CSV,
    and displays the 5-panel visualization dashboard.
    """
    print("\n" + "=" * 65)
    print(f" INSPECTING: {image_path}")
    print("=" * 65)

    if not os.path.exists(image_path):
        print(f"[ERROR] Specified file does not exist: '{image_path}'")
        raise FileNotFoundError(f"File not found: {image_path}")

    # Perform automated CV defect detection
    result = inspect_blister_pack(image_path, reference_image_path=reference_image_path)

    # Print Inspection Outcome
    print(f" Decision         : {result.overall_decision}")
    print(f" Defects Detected : {result.total_defects}")
    print(f" Latency          : {result.processing_time_ms} ms")
    print(f" Summary          : {result.summary_notes}")

    if result.defects:
        print("\n Defect Breakdown:")
        for idx, d in enumerate(result.defects, 1):
            print(f"   [{idx}] Type: {d.defect_type:<18} | Decision: {d.decision} | Confidence: {d.confidence:.2f}")
            print(f"       Details: {d.description}")
            print(f"       Bounding Box: (x={d.bbox[0]}, y={d.bbox[1]}, w={d.bbox[2]}, h={d.bbox[3]})")
    else:
        print("\n All cavities and seal margins passed quality specifications.")

    # Log record to CSV report
    logged_row = log_inspection_result(result)
    print(f"\n[REPORT] Logged to '{DEFAULT_REPORT_PATH}'")

    # Render 5-panel visualization dashboard
    display_five_panel_visualization(image_path, result, save_to_results=True, show_plot=show_ui)

    return result


def run_batch_inspection(dataset_dir: str = "dataset", show_individual_plots: bool = False) -> None:
    """
    Iterates over all blister images across all subfolders in the dataset directory,
    logs results, and outputs executive quality metrics.
    """
    print("\n" + "=" * 65)
    print(f" STARTING BATCH INSPECTION ON: '{dataset_dir}'")
    print("=" * 65)

    image_extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    all_files = []
    for ext in image_extensions:
        all_files.extend(glob.glob(os.path.join(dataset_dir, "**", ext), recursive=True))

    if not all_files:
        print(f"[WARNING] No image files found in '{dataset_dir}'.")
        print("Tip: Run the dataset generator first using Option 3 or `python generate_dataset.py`.")
        return

    print(f"Found {len(all_files)} images across subdirectories. Processing...\n")

    for idx, file_path in enumerate(sorted(all_files), 1):
        rel_path = os.path.relpath(file_path, dataset_dir)
        try:
            result = inspect_blister_pack(file_path)
            log_inspection_result(result)
            status_symbol = "[OK]" if result.overall_decision == "PASS" else "[FAIL]"
            print(f" {idx:2d}/{len(all_files):2d} {status_symbol} {result.overall_decision:<6} | {result.total_defects} defect(s) | {result.processing_time_ms:5.1f} ms | {rel_path}")

            if show_individual_plots:
                display_five_panel_visualization(file_path, result, save_to_results=True, show_plot=True)
        except Exception as e:
            print(f" [ERROR] Failed processing {file_path}: {e}")

    # Output aggregate metrics table
    print_batch_metrics()


def open_file_dialog() -> Optional[str]:
    """Opens a native OS file picker dialog to let the user select a blister pack image."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw() # Hide root Tk window
        root.attributes("-topmost", True)
        file_path = filedialog.askopenfilename(
            title="Select Blister Pack Image for Inspection",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp"), ("All files", "*.*")]
        )
        root.destroy()
        return file_path if file_path else None
    except Exception as e:
        print(f"[FILE DIALOG] Graphical file picker unavailable ({e}). Falling back to terminal input.")
        return None


def launch_browser_app() -> None:
    """Launches the Streamlit browser-based graphical user interface."""
    print("\n" + "=" * 65)
    print(" LAUNCHING PHARMAVISION WEB BROWSER INTERFACE...")
    print("=" * 65)
    print("Starting Streamlit web server. Your browser should open automatically.")
    print("If it does not, navigate to http://localhost:8501 in your browser.")
    print("Press Ctrl+C in this terminal to stop the web server.\n")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\nWeb server stopped.")
    except Exception as e:
        print(f"[ERROR] Failed to launch web server: {e}")


def interactive_menu() -> None:
    """Displays user-friendly terminal menu for beginners."""
    while True:
        print("\n" + "=" * 65)
        print("  AUTOMATED PHARMACEUTICAL BLISTER PACK INSPECTION SYSTEM")
        print("              OpenCV Classical Computer Vision")
        print("=" * 65)
        print(" 1. Launch Web Browser Application (Streamlit GUI)")
        print(" 2. Inspect Single Image (Select via File Dialog)")
        print(" 3. Inspect Single Image (Type File Path manually)")
        print(" 4. Run Batch Inspection on all 'dataset/' folders")
        print(" 5. Generate Synthetic Sample Dataset (10 Realistic Images)")
        print(" 6. View Inspection Summary Report & Metrics")
        print(" 7. Exit Application")
        print("-" * 65)

        choice = input("Enter choice (1-7): ").strip()

        if choice == "1":
            launch_browser_app()

        elif choice == "2":
            selected_file = open_file_dialog()
            if selected_file:
                run_single_inspection(selected_file, show_ui=True)
            else:
                print("[INFO] No file selected.")

        elif choice == "3":
            sample_options = glob.glob(os.path.join("dataset", "**", "*.png"), recursive=True)
            if sample_options:
                print("\nAvailable Sample Images:")
                for i, sample in enumerate(sample_options[:8], 1):
                    print(f"   [{i}] {sample}")
            input_path = input("\nEnter image file path (or sample number): ").strip()

            if input_path.isdigit() and sample_options:
                idx = int(input_path) - 1
                if 0 <= idx < len(sample_options):
                    input_path = sample_options[idx]

            if os.path.exists(input_path):
                run_single_inspection(input_path, show_ui=True)
            else:
                print(f"[ERROR] Invalid path: '{input_path}'")

        elif choice == "4":
            show_vis = input("Display interactive 5-panel window for each image? (y/n, default=n): ").strip().lower() == 'y'
            run_batch_inspection(dataset_dir="dataset", show_individual_plots=show_vis)

        elif choice == "5":
            build_full_dataset()
            print("[SUCCESS] Dataset generated in 'dataset/' subfolders.")

        elif choice == "6":
            print_batch_metrics()

        elif choice == "7":
            print("\nExiting Blister Pack Inspection System. Goodbye!")
            sys.exit(0)

        else:
            print("[ERROR] Invalid choice. Please choose a number between 1 and 7.")


def main():
    parser = argparse.ArgumentParser(
        description="Automated Pharmaceutical Blister Pack Defect Detection System using OpenCV."
    )
    parser.add_argument(
        "--web", "-w",
        action="store_true",
        help="Launch the interactive web browser interface (Streamlit)."
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        help="Path to an individual blister pack image for inspection."
    )
    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="Run batch inspection on all images in the 'dataset/' folder."
    )
    parser.add_argument(
        "--dataset-dir", "-d",
        type=str,
        default="dataset",
        help="Directory path for batch inspection dataset."
    )
    parser.add_argument(
        "--generate-data", "-g",
        action="store_true",
        help="Generate synthetic blister pack images in the dataset folders."
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run in headless mode without popping up Matplotlib GUI windows."
    )
    parser.add_argument(
        "--reference", "-r",
        type=str,
        default=None,
        help="Optional path to a golden reference image for comparison."
    )

    args = parser.parse_args()

    # CLI Mode: Web Interface
    if args.web:
        launch_browser_app()
        return

    # CLI Mode: Generate dataset
    if args.generate_data:
        build_full_dataset(args.dataset_dir)
        return

    # CLI Mode: Single image inspection
    if args.image:
        run_single_inspection(args.image, reference_image_path=args.reference, show_ui=not args.no_gui)
        return

    # CLI Mode: Batch inspection
    if args.batch:
        run_batch_inspection(dataset_dir=args.dataset_dir, show_individual_plots=not args.no_gui)
        return

    # Default to Interactive Menu
    interactive_menu()


if __name__ == "__main__":
    main()
