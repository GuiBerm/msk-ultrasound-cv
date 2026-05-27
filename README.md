# msk-ultrasound-cv

A deep learning pipeline for the ordinal classification of Rheumatoid Arthritis (RA) pathologies using Musculoskeletal (MSK) ultrasound images. 

This project aims to predict structural damage and active inflammation by processing dual-modality imaging (B-Mode and Power Doppler). 

**Key Dataset Characteristics & Challenges:**
* **Multi-Target Ordinal Labels:** Predicting severity scores (0-3) across multiple clinical variables.
* **Dual-Modality Physics:** Handling the distinct clinical differences between grayscale structural imaging and color-flow vascular imaging.
* **Sparse Medical Data:** Managing missing labels, clinical protocol violations, and anatomical constraints using masked learning strategies.
* **Multi-Instance Anatomy:** Addressing overlapping clinical evaluations within single image frames (e.g., complex wrist joints).