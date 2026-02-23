# CrabNet: Compositionally-restricted attention-based Network

## Overview

CrabNet (Compositionally-restricted attention-based Network) is a deep learning model that predicts the bandgaps of crystal materials based on their chemical composition alone. The model learns high-dimensional representations of chemical elements and combines them using attention mechanisms to make accurate predictions without requiring explicit structural information.

## Algorithm

CrabNet employs a two-stage approach:

1. **Element Representation Learning**: The model learns latent representations of elements, capturing chemical properties and trends from large materials databases.

2. **Composition-Based Prediction**: Given a chemical composition (A_x B_y C_z...), the model:
   - Extracts learned element representations
   - Aggregates them using trainable element fractions as weights
   - Passes the composition embedding through transformer-based attention layers
   - Outputs a predicted bandgap value

Key advantages:
- **Data-efficient**: Requires only composition, not full crystal structure
- **Interpretable**: Learned element embeddings correlate with known elemental properties
- **Generalizable**: Trained on diverse materials (halide perovskites, chalcogenides, etc.)

## Usage in this Project

In this pipeline, CrabNet is used to:
- Predict bandgaps for candidate chalcogenide perovskite compositions
- Serve as a key property descriptor for multi-objective optimization
- Support the experimental plausibility assessment and sustainability ranking

See [`4_bandgap_prediction.ipynb`](../../notebooks/4_bandgap_prediction.ipynb) for training, evaluation, and inference workflows.

## References

**Original Paper:**
> Wang, A. Y.-T., et al. *"Element representation learned by feed forward structure encoding with an expanded embedding space."* **npj Computational Materials** 7, 77 (2021).
>
> DOI: [10.1038/s41524-021-00545-1](https://doi.org/10.1038/s41524-021-00545-1)

**Original Repository:**
> [CrabNet GitHub Repository](https://github.com/anthony-wang/crabnet)

## Credits

CrabNet was developed by Anthony Y.-T. Wang and team at UC Berkeley. This integration adapts their model for bandgap prediction in chalcogenide perovskites.
