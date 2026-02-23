# GCNN: Graph Convolutional Neural Networks for Relative Synthesizability Prediction

## Overview

GCNN (Graph Convolutional Neural Network) is a deep learning architecture designed to predict the synthesizability of crystal materials based on their crystal structures. Unlike composition-based approaches, GCNN leverages the full structural information encoded in crystal graphs to quantify the likelihood that a proposed composition can be experimentally synthesized.

## Algorithm

GCNN employs a multi-step approach combining transfer learning and positive-unlabeled (PU) learning:

1. **Structure Encoding**: The crystal structure is represented as a graph where:
   - Nodes represent atoms in the unit cell
   - Edges represent interatomic connections based on distance thresholds
   - Node and edge features encode elemental properties and geometric information

2. **Graph Convolution**: Message-passing layers aggregate information across the crystal graph:
   - Local structural patterns are learned through convolutional operations
   - Long-range structural features are captured through multiple graph convolution layers
   - Attention mechanisms weight the importance of different atomic environments

3. **Transfer Learning**: The model is pre-trained on large materials databases (e.g., Materials Project):
   - Initial weights capture general crystal chemistry principles
   - Fine-tuning on perovskite-specific data improves prediction accuracy

4. **Positive-Unlabeled Learning**: Handles the scarcity of experimentally validated data:
   - Known synthesizable materials (positive set)
   - Unlabeled hypothetical materials (pseudo-labeled as negative with confidence adjustment)
   - Learns decision boundaries that generalize to unseen compositions

5. **Synthesizability Score**: For each structure, the model outputs a "crystal-likeness" score indicating the probability of successful synthesis.

## Usage in this Project

In this pipeline, GCNN is used to:
- Assess the experimental plausibility of CrystaLLM-generated perovskite-type structures
- Generate crystal-likeness scores for all candidate compounds
- Rank candidates by synthesizability likelihood
- Combine synthesizability with sustainability metrics for final prioritization

See [`3_Experimental_likelihood.ipynb`](../../notebooks/3_Experimental_likelihood.ipynb) for structure evaluation and synthesizability scoring workflows.

## Key Advantages

- **Structure-aware**: Incorporates full 3D crystal geometry, not just composition
- **Transfer-learning ready**: Leverages pre-trained weights from large databases
- **Uncertainty-aware**: Handles imbalanced datasets with PU learning
- **Interpretable embeddings**: Learned representations correlate with known synthesizability factors

## References

**Original Paper:**
> Gu, G. H., Jang, J., Noh, H., Jang, W., Walsh, A., & Jung, Y. (2022). *"Perovskite synthesizability using graph neural networks."* **npj Computational Materials** 8, 71.
>
> DOI: [10.1038/s41524-022-00757-z](https://doi.org/10.1038/s41524-022-00757-z)

**Original Repository:**
> [PerovskiteSynthesizability_Manuscript2021](https://github.com/kaist-amsg/PerovskiteSynthesizability_Manuscript2021)

## Credits

The GCNN model and synthesizability assessment methodology were developed by Geun Ho Gu, Jidon Jang, Hyunwoo Noh, and colleagues at KAIST in collaboration with Aron Walsh (Imperial College London) and Yousung Jung (KAIST). This integration adapts their model for evaluating the experimental plausibility of chalcogenide perovskite structures.
