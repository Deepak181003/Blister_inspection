"""
generate_dataset.py
===================
Synthetic Pharmaceutical Blister Pack Image Generator.

Generates realistic sample blister pack images for training, testing,
and evaluating the defect detection computer vision pipeline.

Image Categories Created:
  1. normal/          - Perfect blister packs with 10 intact tablets.
  2. missing_capsule/ - Blister pack with 1 or 2 empty cavities.
  3. cracked_pill/    - Blister pack with fractured / cracked tablets.
  4. broken_seal/     - Blister pack with foil tears or punctured seal margins.
  5. deformed_cavity/ - Blister pack with crushed / warped cavity pockets.
"""

import os
import cv2
import numpy as np
from typing import Tuple


def draw_blister_card(width: int = 800, height: int = 480) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Creates base canvas with dark conveyor background and metallic blister foil card."""
    # Dark industrial conveyor belt background
    canvas = np.full((height, width, 3), (38, 40, 44), dtype=np.uint8)

    # Card dimensions (centered)
    card_w = 680
    card_h = 360
    card_x = (width - card_w) // 2
    card_y = (height - card_h) // 2

    # Draw aluminium foil blister card (silver/metallic base)
    cv2.rectangle(canvas, (card_x, card_y), (card_x + card_w, card_y + card_h), (210, 215, 220), -1)

    # Knurled heat-seal texture (fine grid lines on aluminium packaging)
    for gx in range(card_x, card_x + card_w, 12):
        cv2.line(canvas, (gx, card_y), (gx, card_y + card_h), (195, 200, 205), 1)
    for gy in range(card_y, card_y + card_h, 12):
        cv2.line(canvas, (card_x, gy), (card_x + card_w, gy), (195, 200, 205), 1)

    # Beveled card border
    cv2.rectangle(canvas, (card_x, card_y), (card_x + card_w, card_y + card_h), (160, 165, 170), 2)

    return canvas, (card_x, card_y, card_w, card_h)


def draw_cavity(image: np.ndarray,
                center: Tuple[int, int],
                radius_x: int = 40,
                radius_y: int = 40,
                deformed: bool = False) -> None:
    """Draws 3D thermoformed blister pocket recess with depth shading."""
    cx, cy = center
    rx, ry = (int(radius_x * 1.35), int(radius_y * 0.70)) if deformed else (radius_x, radius_y)

    # Outer recess ring (shadow on top/left, highlight on bottom/right)
    cv2.ellipse(image, (cx, cy), (rx + 4, ry + 4), 0, 0, 360, (140, 145, 150), 2, cv2.LINE_AA)
    # Pocket floor
    cv2.ellipse(image, (cx, cy), (rx, ry), 0, 0, 360, (175, 180, 185), -1, cv2.LINE_AA)
    # Inner shadow gradient
    cv2.ellipse(image, (cx - 2, cy - 2), (rx - 2, ry - 2), 0, 180, 360, (135, 140, 145), 2, cv2.LINE_AA)


def draw_tablet(image: np.ndarray,
                center: Tuple[int, int],
                radius: int = 32,
                color: Tuple[int, int, int] = (245, 245, 248),
                cracked: bool = False,
                deformed: bool = False) -> None:
    """Draws a pharmaceutical tablet with lighting bevel, optionally cracked or deformed."""
    cx, cy = center
    rx, ry = (int(radius * 1.3), int(radius * 0.65)) if deformed else (radius, radius)

    # Tablet shadow
    cv2.ellipse(image, (cx + 2, cy + 3), (rx, ry), 0, 0, 360, (120, 125, 130), -1, cv2.LINE_AA)
    # Main tablet body
    cv2.ellipse(image, (cx, cy), (rx, ry), 0, 0, 360, color, -1, cv2.LINE_AA)
    # Tablet rim / bevel ring
    cv2.ellipse(image, (cx, cy), (rx - 3, ry - 3), 0, 0, 360, (215, 218, 222), 1, cv2.LINE_AA)

    # Score line / debossing in center of tablet
    if not cracked and not deformed:
        cv2.line(image, (cx, cy - int(ry * 0.6)), (cx, cy + int(ry * 0.6)), (210, 212, 215), 1, cv2.LINE_AA)

    # Cracked pill defect: draw sharp, jagged fracture line across tablet body
    if cracked:
        # Irregular jagged crack points across the tablet surface
        pts = np.array([
            [cx - int(rx * 0.75), cy - int(ry * 0.60)],
            [cx - int(rx * 0.25), cy - int(ry * 0.15)],
            [cx - int(rx * 0.10), cy + 2],
            [cx + int(rx * 0.35), cy - 4],
            [cx + int(rx * 0.78), cy + int(ry * 0.55)]
        ], np.int32)
        cv2.polylines(image, [pts], False, (40, 40, 45), 3, cv2.LINE_AA)
        
        # Secondary hairline fracture branch
        branch_pts = np.array([
            [cx - int(rx * 0.10), cy + 2],
            [cx + int(rx * 0.15), cy + int(ry * 0.50)],
            [cx + int(rx * 0.40), cy + int(ry * 0.70)]
        ], np.int32)
        cv2.polylines(image, [branch_pts], False, (45, 45, 50), 2, cv2.LINE_AA)
        
        # Missing chip at the corner
        chip_pt = (cx - int(rx * 0.8), cy - int(ry * 0.6))
        cv2.circle(image, chip_pt, 7, (140, 145, 150), -1, cv2.LINE_AA)


def generate_blister_image(defect_type: str = 'normal',
                           defect_index: int = 1) -> np.ndarray:
    """
    Generates a full synthetic blister pack image (2 rows x 5 columns = 10 cavities).

    Args:
        defect_type: 'normal', 'missing_capsule', 'cracked_pill', 'broken_seal', or 'deformed_cavity'.
        defect_index: Variation index for slight differences.
    """
    image, (card_x, card_y, card_w, card_h) = draw_blister_card()

    # Calculate 2x5 cavity coordinates
    rows = 2
    cols = 5
    margin_x = int(card_w * 0.08)
    margin_y = int(card_h * 0.08)
    active_w = card_w - (2 * margin_x)
    active_h = card_h - (2 * margin_y)
    
    cell_w = active_w / cols
    cell_h = active_h / rows

    cavity_num = 1
    for r in range(rows):
        for c in range(cols):
            cx = int(card_x + margin_x + (c + 0.5) * cell_w)
            cy = int(card_y + margin_y + (r + 0.5) * cell_h)

            # Determine if this specific cavity should have a defect
            is_missing = False
            is_cracked = False
            is_deformed = False

            if defect_type == 'missing_capsule':
                # Empty cavity at pocket #3 (or #7 for variant 2)
                target_cavity = 3 if defect_index == 1 else 7
                if cavity_num == target_cavity:
                    is_missing = True

            elif defect_type == 'cracked_pill':
                # Cracked tablet at pocket #4 (or #2 for variant 2)
                target_cavity = 4 if defect_index == 1 else 2
                if cavity_num == target_cavity:
                    is_cracked = True

            elif defect_type == 'deformed_cavity':
                # Deformed pocket at pocket #8 (or #5 for variant 2)
                target_cavity = 8 if defect_index == 1 else 5
                if cavity_num == target_cavity:
                    is_deformed = True

            # Draw thermoformed pocket recess
            draw_cavity(image, (cx, cy), deformed=is_deformed)

            # Draw tablet unless it's missing
            if not is_missing:
                draw_tablet(image, (cx, cy), cracked=is_cracked, deformed=is_deformed)

            cavity_num += 1

    # Inject broken seal defect if requested
    if defect_type == 'broken_seal':
        if defect_index == 1:
            # Tear notch along the top-right foil flange
            notch_x = card_x + card_w - 60
            notch_y = card_y
            notch_pts = np.array([
                [notch_x - 30, notch_y],
                [notch_x, notch_y + 35],
                [notch_x + 30, notch_y]
            ], np.int32)
            cv2.fillPoly(image, [notch_pts], (38, 40, 44), cv2.LINE_AA)
            cv2.polylines(image, [notch_pts], False, (20, 20, 20), 2, cv2.LINE_AA)
        else:
            # Puncture / rupture hole along the bottom seal margin
            punc_x = card_x + int(card_w * 0.45)
            punc_y = card_y + card_h - 18
            cv2.ellipse(image, (punc_x, punc_y), (35, 14), 0, 0, 360, (38, 40, 44), -1, cv2.LINE_AA)
            cv2.ellipse(image, (punc_x, punc_y), (36, 15), 0, 0, 360, (15, 15, 18), 2, cv2.LINE_AA)

    return image


def build_full_dataset(dataset_dir: str = "dataset") -> None:
    """Generates a complete suite of synthetic test images across all categories."""
    categories = {
        "normal": "normal",
        "missing_capsule": "missing_capsule",
        "cracked_pill": "cracked_pill",
        "broken_seal": "broken_seal",
        "deformed_cavity": "deformed_cavity"
    }

    print(f"[DATASET] Creating test dataset in '{dataset_dir}'...")

    total_created = 0
    for folder_name, defect_key in categories.items():
        folder_path = os.path.join(dataset_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        # Generate 2 distinct sample images per category
        for idx in [1, 2]:
            img = generate_blister_image(defect_type=defect_key, defect_index=idx)
            filename = f"blister_{folder_name}_{idx:02d}.png"
            full_path = os.path.join(folder_path, filename)
            cv2.imwrite(full_path, img)
            total_created += 1
            print(f"  + Generated: {full_path}")

    print(f"[DATASET] Successfully created {total_created} sample images.")


if __name__ == "__main__":
    build_full_dataset()
