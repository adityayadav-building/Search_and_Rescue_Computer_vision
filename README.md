

# Search & Rescue Computer Vision Pipeline

A Python-based computer vision system designed to process UAV imagery for autonomous search and rescue operations. This pipeline segments terrain, identifies casualties and rescue camps, calculates medical priorities, and optimizes rescue allocation based on capacity constraints.

## 🚀 Features

* **Terrain Segmentation:** Automatically separates Land vs. Ocean using custom HSV masking and morphological operations.
* **Object Detection:** Identifies casualties (Shapes: Triangle, Square, Star) and Rescue Camps (Circles) using contour analysis.
* **Priority System:** Calculates a priority score for each casualty based on:
    * **Shape:** (Star=Child, Triangle=Elderly, Square=Adult)
    * **Color:** (Red=Severe, Yellow=Mild, Green=Safe)
* **Rescue Optimization:** Implements a greedy algorithm to assign high-priority casualties to the nearest available camp, respecting specific camp capacity limits.
* **Batch Processing:** Automatically processes a full dataset of images and outputs a ranked list based on the "Rescue Efficiency Ratio" ($P_r$).

## 🛠️ Tech Stack

* **Language:** Python 3.11
* **Libraries:**
    * `OpenCV` (cv2): Image processing, contour detection, and visualization.
    * `NumPy`: Matrix operations and masking.
    * `Math`: Euclidean distance calculations.
    * `OS`: File handling for batch automation.

## 📂 Project Structure

```text
├── main.py                  # The core script containing the entire pipeline
├── HSV_COLOUR_DETECTOR.py   # Utility script with Trackbars for calibrating HSV values
├── task_images/             # Folder containing input images (1.png, 2.png, etc.)
└── README.md                # Project documentation
