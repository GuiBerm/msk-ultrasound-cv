# MSK Ultrasound CV — Rheumatoid Arthritis Ordinal Scoring Pipeline
"""
Three-model architecture for MSK ultrasound pathology scoring:
  - B-Mode Structural:  eg_sinovial (0-3) + bone_erosion (0-1)
  - Doppler Vascular:   pd_sinovial (0-3)
  - QA Gatekeeper:      joint_type (5 merged classes, geometry-only)

Uses CORN (Conditional Ordinal Regression) with masked multi-task loss for
scoring models, and CrossEntropy for the QA classifier.  Joint-type embedding
conditioning, modality embedding for QA, and pre-computed stratified group
k-folds throughout.
"""
