# MSK Ultrasound CV — Rheumatoid Arthritis Ordinal Scoring Pipeline
"""
Two-model architecture for MSK ultrasound pathology scoring:
  - B-Mode Structural:  eg_sinovial (0-3) + bone_erosion (0-1)
  - Doppler Vascular:   pd_sinovial (0-3)

Uses CORN (Conditional Ordinal Regression) with masked multi-task loss,
joint-type embedding conditioning, and pre-computed stratified group k-folds.
"""
