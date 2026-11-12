"""
detection.py
============
Core Defect Detection and Classification Engine for Pharmaceutical Blister Packs.

Defects Detected:
  1. Missing Capsules / Tablets (Empty Cavity)
  2. Cracked or Broken Tablets (Fracture lines, Chipping, Low Circularity)
  3. Damaged or Broken Blister Seals (Outer foil tears, Punctures, Seal ruptures)
  4. Deformed Blister Cavities (Crushed, elongated, or warped thermoformed pockets)

Classification Decisions:
  - PASS: All cavities and seals are within strict tolerance limits.
  - REJECT: One or more definitive defects detected.
  - REVIEW: Uncertain / borderline results requiring manual human inspection.

Techniques Implemented:
  - Contour analysis (Area, Perimeter, Aspect Ratio, Solidity, Circularity)
  - Region of Interest (ROI) extraction and cavity-level isolation
  - Edge density analysis inside the tablet body (for hairline crack detection)
  - Perimeter seal contour & convexity defect analysis (for foil tear detection)
  - Reference profile comparison (comparing cavity metrics against golden master)
"""

import time
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
from preprocessing import (
    load_image,
    to_grayscale,
    apply_gaussian_blur,
    apply_adaptive_threshold,
    apply_canny_edge,
    apply_morphology
)


@dataclass
class CavityDefect:
    """Stores information about a single detected defect in a blister cavity or seal."""
    cavity_id: int
    defect_type: str                # 'MISSING_TABLET', 'CRACKED_TABLET', 'DEFORMED_CAVITY', 'BROKEN_SEAL'
    bbox: Tuple[int, int, int, int] # (x, y, w, h)
    confidence: float               # 0.0 to 1.0 confidence score
    decision: str                   # 'REJECT' or 'REVIEW'
    description: str                # Human-readable explanation


@dataclass
class InspectionResult:
    """Stores the complete inspection outcome for one blister pack image."""
    image_filename: str
    overall_decision: str           # 'PASS', 'REJECT', or 'REVIEW'
    total_defects: int
    defects: List[CavityDefect]
    cavity_rois: List[Tuple[int, int, int, int]]
    processing_time_ms: float
    annotated_image: np.ndarray
    summary_notes: str


class BlisterPackReference:
    """
    Reference Profile (Golden Master) for blister pack specifications.

    Why this is used:
      - In industrial pharma quality control, camera systems are configured
        with nominal physical dimensions (e.g. tablet radius, cavity spacing).
      - Comparing each incoming blister against the nominal reference allows
        detection of missing pills, deformed shapes, and missing foil areas.
    """
    def __init__(self,
                 grid_rows: int = 2,
                 grid_cols: int = 5,
                 expected_cavity_count: int = 10,
                 nominal_aspect_ratio: float = 1.0,
                 max_aspect_ratio_deviation: float = 0.35,
                 min_fill_ratio: float = 0.15,
                 review_fill_ratio: float = 0.28,
                 min_solidity: float = 0.90,
                 max_internal_edge_ratio: float = 0.025,
                 min_circularity: float = 0.65):
        """
        Args:
            grid_rows (int): Number of cavity rows on the blister card.
            grid_cols (int): Number of cavity columns on the blister card.
            expected_cavity_count (int): Total expected cavities (rows * cols).
            nominal_aspect_ratio (float): Width / Height ratio for normal cavity (1.0 for round).
            max_aspect_ratio_deviation (float): Maximum allowed difference from nominal aspect ratio.
            min_fill_ratio (float): Minimum pill area fraction inside cavity to be considered present.
            review_fill_ratio (float): Borderline fill ratio threshold requiring human review.
            min_solidity (float): Minimum contour solidity (Area / ConvexHullArea) for unbroken pill.
            max_internal_edge_ratio (float): Maximum allowed edge pixel density inside pill body.
            min_circularity (float): 4*pi*Area / Perimeter^2 (1.0 = perfect circle).
        """
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.expected_cavity_count = expected_cavity_count
        self.nominal_aspect_ratio = nominal_aspect_ratio
        self.max_aspect_ratio_deviation = max_aspect_ratio_deviation
        self.min_fill_ratio = min_fill_ratio
        self.review_fill_ratio = review_fill_ratio
        self.min_solidity = min_solidity
        self.max_internal_edge_ratio = max_internal_edge_ratio
        self.min_circularity = min_circularity


def calculate_circularity(area: float, perimeter: float) -> float:
    """
    Computes shape circularity (also known as isoperimetric quotient):
        Circularity = (4 * pi * Area) / (Perimeter^2)

    Why this matters:
      - A perfect geometric circle has a circularity of 1.0.
      - Smooth tablets score ~0.85 - 0.95.
      - Cracked, chipped, or indented pills score significantly lower (< 0.70).
    """
    if perimeter <= 0 or area <= 0:
        return 0.0
    return float((4.0 * np.pi * area) / (perimeter * perimeter))


def calculate_solidity(contour: np.ndarray) -> float:
    """
    Computes contour solidity:
        Solidity = Contour Area / Convex Hull Area

    Why this matters:
      - The 'Convex Hull' is like stretching a rubber band around the outer pill contour.
      - An unbroken, intact round tablet is completely convex, so its Solidity is close to 1.0.
      - If a tablet is cracked or chipped, the contour has notches that pull away from the
        convex hull, causing Solidity to drop.
    """
    area = cv2.contourArea(contour)
    if area <= 0:
        return 0.0
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return 0.0
    return float(area) / float(hull_area)


def extract_pack_boundary(image: np.ndarray) -> Tuple[Tuple[int, int, int, int], Optional[np.ndarray], np.ndarray]:
    """
    Locates the outer rectangular foil border of the blister pack card.

    Uses adaptive dynamic midpoint thresholding between foreground and background
    to separate the metallic blister card from conveyor background reliably.

    Args:
        image (np.ndarray): Original BGR image.

    Returns:
        Tuple: ((x, y, w, h) bounding box, card contour, binary card threshold mask).
    """
    gray = to_grayscale(image)
    blurred = apply_gaussian_blur(gray, (5, 5), 1.0)
    
    # Calculate midpoint between minimum background and maximum card brightness
    min_val, max_val, _, _ = cv2.minMaxLoc(blurred)
    mid_thresh = int((min_val + max_val) / 2.0)
    
    # Binarize card
    _, thresh = cv2.threshold(blurred, mid_thresh, 255, cv2.THRESH_BINARY)
    thresh_clean = apply_morphology(thresh, 'close', (9, 9), iterations=2)
    
    contours, _ = cv2.findContours(thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (0, 0, image.shape[1], image.shape[0]), None, thresh_clean
        
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Sanity check: must occupy at least 20% of image canvas
    img_area = image.shape[0] * image.shape[1]
    if (w * h) < (0.20 * img_area):
        return (0, 0, image.shape[1], image.shape[0]), None, thresh_clean
        
    return (x, y, w, h), largest_contour, thresh_clean


def inspect_blister_seals(image: np.ndarray,
                          pack_bbox: Tuple[int, int, int, int],
                          pack_contour: Optional[np.ndarray],
                          card_thresh: np.ndarray) -> List[CavityDefect]:
    """
    Inspects the blister foil seal margins and perimeter for tears, punctures, or breaks.

    Defect criteria:
      1. Outer foil tear / notch: Detected via Convexity Defects along outer pack contour (depth > 18px).
      2. Puncture hole: Detected via internal hole contours in the outer seal margin.
    """
    defects: List[CavityDefect] = []
    px, py, pw, ph = pack_bbox
    
    # Margin where outer heat seal exists (8% of pack dimension)
    margin_x = int(pw * 0.08)
    margin_y = int(ph * 0.08)

    # 1. Outer Foil Tear Inspection (Convexity Defects)
    if pack_contour is not None and len(pack_contour) >= 5:
        hull = cv2.convexHull(pack_contour, returnPoints=False)
        if hull is not None and len(hull) > 3:
            try:
                defect_pts = cv2.convexityDefects(pack_contour, hull)
                if defect_pts is not None:
                    for i in range(defect_pts.shape[0]):
                        s, e, f, d = defect_pts[i, 0]
                        depth = d / 256.0 # Distance in pixels from convex hull
                        
                        # An indentation > 18 pixels in the outer foil indicates a torn edge
                        if depth > 18.0:
                            far_pt = tuple(pack_contour[f][0])
                            bx = max(0, far_pt[0] - 25)
                            by = max(0, far_pt[1] - 25)
                            bw = 50
                            bh = 50
                            defects.append(CavityDefect(
                                cavity_id=991,
                                defect_type='BROKEN_SEAL',
                                bbox=(bx, by, bw, bh),
                                confidence=0.94,
                                decision='REJECT',
                                description=f"Foil seal tear / edge notch detected (tear depth: {depth:.1f}px)"
                            ))
                            break
            except Exception:
                pass

    # 2. Puncture / Rupture Inspection in Outer Seal Margins
    # Find all contours including internal holes using RETR_CCOMP
    cnts_all, _ = cv2.findContours(card_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts_all:
        hole_area = cv2.contourArea(c)
        # Significant puncture hole
        if 800 < hole_area < 5000:
            bx, by, bw, bh = cv2.boundingRect(c)
            # Check if this hole lies inside the outer seal margin (border zone)
            in_top_seal = (py <= by <= py + margin_y + 15)
            in_bottom_seal = (py + ph - margin_y - 25 <= by <= py + ph)
            in_left_seal = (px <= bx <= px + margin_x + 15)
            in_right_seal = (px + pw - margin_x - 25 <= bx <= px + pw)

            if in_top_seal or in_bottom_seal or in_left_seal or in_right_seal:
                # Avoid duplicate with tear defect
                overlap = any(abs(bx - d.bbox[0]) < 40 and abs(by - d.bbox[1]) < 40 for d in defects)
                if not overlap:
                    defects.append(CavityDefect(
                        cavity_id=992,
                        defect_type='BROKEN_SEAL',
                        bbox=(bx, by, bw, bh),
                        confidence=0.92,
                        decision='REJECT',
                        description=f"Seal breach / puncture hole detected in blister flange (Area: {hole_area:.0f}px)"
                    ))

    return defects


def locate_cavity_rois(image: np.ndarray,
                       pack_bbox: Tuple[int, int, int, int],
                       reference: BlisterPackReference) -> List[Tuple[int, int, int, int]]:
    """
    Identifies the Regions of Interest (ROIs) for every individual blister pocket.

    Uses calibrated grid geometry derived from the localized blister card.
    Focuses a square/rectangular pocket window around each cavity center.

    Args:
        image (np.ndarray): Original BGR image.
        pack_bbox: (x, y, w, h) of the outer blister card.
        reference (BlisterPackReference): Reference layout specs.

    Returns:
        List[Tuple[int, int, int, int]]: List of (x, y, w, h) bounding boxes for all cavities.
    """
    px, py, pw, ph = pack_bbox

    # Active internal area excluding outer heat-seal margins
    margin_x = int(pw * 0.08)
    margin_y = int(ph * 0.08)
    active_w = pw - (2 * margin_x)
    active_h = ph - (2 * margin_y)

    cell_w = active_w / reference.grid_cols
    cell_h = active_h / reference.grid_rows

    # Size of the pocket ROI centered within each grid cell
    pocket_size = int(min(cell_w, cell_h) * 0.78)
    half_size = pocket_size // 2

    cavity_rois: List[Tuple[int, int, int, int]] = []

    for r in range(reference.grid_rows):
        for c in range(reference.grid_cols):
            cx = int(px + margin_x + (c + 0.5) * cell_w)
            cy = int(py + margin_y + (r + 0.5) * cell_h)

            x = max(0, min(cx - half_size, image.shape[1] - pocket_size))
            y = max(0, min(cy - half_size, image.shape[0] - pocket_size))
            w = pocket_size
            h = pocket_size

            cavity_rois.append((x, y, w, h))

    return cavity_rois


def inspect_single_cavity(cavity_id: int,
                          image: np.ndarray,
                          roi_bbox: Tuple[int, int, int, int],
                          reference: BlisterPackReference) -> Tuple[Optional[CavityDefect], Dict[str, Any]]:
    """
    Inspects an individual blister cavity pocket for:
      - Missing tablet / capsule
      - Cracked or fractured tablet
      - Deformed thermoformed cavity pocket

    Args:
        cavity_id (int): Sequential index of cavity (1 to N).
        image (np.ndarray): Full BGR image.
        roi_bbox (Tuple[int, int, int, int]): (x, y, w, h) of the cavity ROI.
        reference (BlisterPackReference): Golden reference thresholds.

    Returns:
        Tuple: (CavityDefect or None, Diagnostic metrics dictionary).
    """
    x, y, w, h = roi_bbox
    roi_bgr = image[y:y+h, x:x+w]
    roi_area = float(w * h)

    if roi_bgr.size == 0 or w < 10 or h < 10:
        return None, {"error": "Invalid ROI size"}

    gray_roi = to_grayscale(roi_bgr)
    blurred_roi = apply_gaussian_blur(gray_roi, (5, 5), 1.0)

    # Threshold for tablet core pixels (bright pharmaceutical tablet material > 230)
    pill_mask = (blurred_roi > 225).astype(np.uint8) * 255
    pill_pixel_count = int(cv2.countNonZero(pill_mask))
    fill_ratio = float(pill_pixel_count) / roi_area

    metrics: Dict[str, Any] = {
        "cavity_id": cavity_id,
        "fill_ratio": round(fill_ratio, 3),
        "aspect_ratio": 1.0,
        "solidity": 1.0,
        "circularity": 1.0,
        "internal_edge_ratio": 0.0
    }

    # -------------------------------------------------------------
    # 1. MISSING TABLET ANALYSIS
    # -------------------------------------------------------------
    # If tablet pixel area is below empty threshold
    if fill_ratio < reference.min_fill_ratio:
        defect = CavityDefect(
            cavity_id=cavity_id,
            defect_type="MISSING_TABLET",
            bbox=roi_bbox,
            confidence=0.98,
            decision="REJECT",
            description=f"Cavity #{cavity_id}: Missing tablet (Fill ratio: {fill_ratio:.2f} < {reference.min_fill_ratio:.2f})"
        )
        return defect, metrics

    # Borderline fill ratio requiring human review
    if fill_ratio < reference.review_fill_ratio:
        defect = CavityDefect(
            cavity_id=cavity_id,
            defect_type="MISSING_TABLET",
            bbox=roi_bbox,
            confidence=0.68,
            decision="REVIEW",
            description=f"Cavity #{cavity_id}: Borderline tablet volume (Fill ratio: {fill_ratio:.2f})"
        )
        return defect, metrics

    # -------------------------------------------------------------
    # 2. TABLET CONTOUR & GEOMETRY EXTRACTION
    # -------------------------------------------------------------
    contours, _ = cv2.findContours(pill_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, metrics

    primary_contour = max(contours, key=cv2.contourArea)
    tablet_area = cv2.contourArea(primary_contour)
    perimeter = cv2.arcLength(primary_contour, True)

    bx, by, bw, bh = cv2.boundingRect(primary_contour)
    aspect_ratio = float(bw) / float(bh) if bh > 0 else 1.0
    circularity = calculate_circularity(tablet_area, perimeter)
    solidity = calculate_solidity(primary_contour)

    metrics["aspect_ratio"] = round(aspect_ratio, 3)
    metrics["circularity"] = round(circularity, 3)
    metrics["solidity"] = round(solidity, 3)

    # -------------------------------------------------------------
    # 3. CRACKED OR BROKEN TABLET ANALYSIS
    # -------------------------------------------------------------
    # Check interior Canny edges inside the tablet body
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded_mask = cv2.erode(pill_mask, erode_kernel, iterations=1)
    eroded_area = cv2.countNonZero(eroded_mask)

    roi_edges = apply_canny_edge(blurred_roi, low_threshold=50, high_threshold=140)
    interior_edges = cv2.bitwise_and(roi_edges, roi_edges, mask=eroded_mask)
    interior_edge_count = cv2.countNonZero(interior_edges)

    internal_edge_ratio = (interior_edge_count / float(eroded_area)) if eroded_area > 0 else 0.0
    metrics["internal_edge_ratio"] = round(internal_edge_ratio, 4)

    is_fractured = internal_edge_ratio > reference.max_internal_edge_ratio
    is_chipped = (solidity < reference.min_solidity) or (circularity < reference.min_circularity)

    if is_fractured or is_chipped:
        defect = CavityDefect(
            cavity_id=cavity_id,
            defect_type="CRACKED_TABLET",
            bbox=roi_bbox,
            confidence=0.94,
            decision="REJECT",
            description=f"Cavity #{cavity_id}: Cracked / broken tablet (Solidity: {solidity:.3f}, Circ: {circularity:.2f})"
        )
        return defect, metrics

    # -------------------------------------------------------------
    # 4. DEFORMED CAVITY ANALYSIS
    # -------------------------------------------------------------
    aspect_ratio_deviation = abs(aspect_ratio - reference.nominal_aspect_ratio)

    if aspect_ratio_deviation > reference.max_aspect_ratio_deviation:
        defect = CavityDefect(
            cavity_id=cavity_id,
            defect_type="DEFORMED_CAVITY",
            bbox=roi_bbox,
            confidence=0.93,
            decision="REJECT",
            description=f"Cavity #{cavity_id}: Deformed pocket geometry (Aspect ratio: {aspect_ratio:.2f}, Deviation: {aspect_ratio_deviation:.2f})"
        )
        return defect, metrics

    # Borderline crack / surface defect requiring review
    if internal_edge_ratio > (reference.max_internal_edge_ratio * 0.70):
        defect = CavityDefect(
            cavity_id=cavity_id,
            defect_type="CRACKED_TABLET",
            bbox=roi_bbox,
            confidence=0.62,
            decision="REVIEW",
            description=f"Cavity #{cavity_id}: Borderline surface irregularity (Internal edges: {internal_edge_ratio:.3f})"
        )
        return defect, metrics

    # Cavity is PASS
    return None, metrics


def annotate_defects(image: np.ndarray,
                     cavity_rois: List[Tuple[int, int, int, int]],
                     defects: List[CavityDefect],
                     overall_decision: str) -> np.ndarray:
    """
    Renders high-contrast bounding boxes, cavity status labels, and inspection banners.

    Color Coding:
      - RED   (0, 0, 255):   Definite Defect (REJECT)
      - YELLOW(0, 255, 255): Borderline condition (REVIEW)
      - GREEN (0, 255, 0):   Normal passing cavity (PASS)

    Args:
        image (np.ndarray): Original BGR image.
        cavity_rois (List[Tuple[int, int, int, int]]): All cavity bounding boxes.
        defects (List[CavityDefect]): List of detected defects.
        overall_decision (str): 'PASS', 'REJECT', or 'REVIEW'.

    Returns:
        np.ndarray: Fully annotated BGR visualization image.
    """
    annotated = image.copy()
    defect_map = {d.cavity_id: d for d in defects if d.cavity_id < 900}

    # 1. Draw cavity boxes
    for idx, (x, y, w, h) in enumerate(cavity_rois, start=1):
        if idx in defect_map:
            defect = defect_map[idx]
            if defect.decision == 'REJECT':
                color = (0, 0, 255)       # Red for REJECT
                label = f"#{idx} REJECT: {defect.defect_type.replace('_', ' ')}"
            else:
                color = (0, 255, 255)     # Yellow for REVIEW
                label = f"#{idx} REVIEW: {defect.defect_type.replace('_', ' ')}"

            # Thick bounding box for defects
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)

            # Label badge
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(annotated, (x, max(0, y - 20)), (x + tw + 6, y), color, -1)
            cv2.putText(annotated, label, (x + 3, max(12, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            # Green box for PASS cavity
            color = (0, 255, 0)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            cv2.putText(annotated, f"#{idx} PASS", (x + 4, y + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    # 2. Draw seal damage defects (cavity_id >= 900)
    for defect in defects:
        if defect.cavity_id >= 900:
            x, y, w, h = defect.bbox
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 3)
            label = "SEAL DAMAGE"
            cv2.rectangle(annotated, (x, max(0, y - 20)), (x + 115, y), (0, 0, 255), -1)
            cv2.putText(annotated, label, (x + 4, max(12, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # 3. Top Banner with Overall Decision
    banner_color = (0, 160, 0) if overall_decision == 'PASS' else (
        (0, 0, 210) if overall_decision == 'REJECT' else (0, 200, 220)
    )
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 44), banner_color, -1)

    status_text = f"DECISION: {overall_decision}  |  DEFECTS FOUND: {len(defects)}"
    cv2.putText(annotated, status_text, (15, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

    return annotated


def inspect_blister_pack(image_path: str,
                         reference: Optional[BlisterPackReference] = None,
                         reference_image_path: Optional[str] = None) -> InspectionResult:
    """
    Main inspection function that performs complete automated inspection
    on a blister pack image.

    Args:
        image_path (str): File path to the test image.
        reference (Optional[BlisterPackReference]): Golden baseline thresholds.
        reference_image_path (Optional[str]): Optional reference golden master image.

    Returns:
        InspectionResult: Complete inspection outcome with annotated image and metrics.
    """
    start_time = time.perf_counter()

    image = load_image(image_path)

    if reference is None:
        reference = BlisterPackReference()

    # Step 1: Blister card outer boundary & seal inspection
    pack_bbox, pack_contour, card_thresh = extract_pack_boundary(image)
    seal_defects = inspect_blister_seals(image, pack_bbox, pack_contour, card_thresh)

    # Step 2: Locate Cavity ROIs
    cavity_rois = locate_cavity_rois(image, pack_bbox, reference)

    # Step 3: Inspect each cavity
    all_defects: List[CavityDefect] = list(seal_defects)
    cavity_metrics = []

    for idx, roi in enumerate(cavity_rois, start=1):
        defect, metrics = inspect_single_cavity(idx, image, roi, reference)
        cavity_metrics.append(metrics)
        if defect is not None:
            all_defects.append(defect)

    # Optional: Reference Image Comparison (if golden master image provided)
    if reference_image_path is not None:
        try:
            ref_img = load_image(reference_image_path)
            if ref_img.shape != image.shape:
                ref_img = cv2.resize(ref_img, (image.shape[1], image.shape[0]))
            diff = cv2.absdiff(to_grayscale(ref_img), to_grayscale(image))
            mean_diff = float(np.mean(diff))
            if mean_diff > 45.0 and len(all_defects) == 0:
                all_defects.append(CavityDefect(
                    cavity_id=0,
                    defect_type="DEFORMED_CAVITY",
                    bbox=pack_bbox,
                    confidence=0.72,
                    decision="REVIEW",
                    description=f"Significant difference from reference master (Mean diff: {mean_diff:.1f})"
                ))
        except Exception:
            pass

    # Step 4: Overall Decision Logic
    has_reject = any(d.decision == 'REJECT' for d in all_defects)
    has_review = any(d.decision == 'REVIEW' for d in all_defects)

    if has_reject:
        overall_decision = 'REJECT'
    elif has_review:
        overall_decision = 'REVIEW'
    else:
        overall_decision = 'PASS'

    # Step 5: Render Visual Annotations
    annotated = annotate_defects(image, cavity_rois, all_defects, overall_decision)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    summary = f"Inspected {len(cavity_rois)} cavities. Detected {len(all_defects)} defect(s). Decision: {overall_decision} in {elapsed_ms:.2f}ms."

    return InspectionResult(
        image_filename=image_path,
        overall_decision=overall_decision,
        total_defects=len(all_defects),
        defects=all_defects,
        cavity_rois=cavity_rois,
        processing_time_ms=round(elapsed_ms, 2),
        annotated_image=annotated,
        summary_notes=summary
    )
