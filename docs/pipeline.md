# Pipeline

The analysis is organized as a sequential pipeline of Jupyter notebooks. Each step builds on the outputs of the previous one, progressively narrowing the candidate space from all possible ABX₃ compositions down to a ranked shortlist of experimentally viable chalcogenide perovskites.

![Screening pipeline workflow](assets/screening_workflow.png)

## Notebooks

Run the notebooks in order. Each notebook is self-contained and will load the required intermediate data from previous steps.

### Step 0 — Publication Figures

[`0_figures_paper.ipynb`](https://github.com/GarzonDiegoFEUP/chalcogenide-perovskite-screening/blob/main/notebooks/0_figures_paper.ipynb)

Generates all publication-quality figures. Can be run after all other steps are complete, or independently using pre-computed results.

---

### Step 1 — Tolerance Factor & Feature Engineering

[`1_get_SISSO_features.ipynb`](https://github.com/GarzonDiegoFEUP/chalcogenide-perovskite-screening/blob/main/notebooks/1_get_SISSO_features.ipynb)

- Load and normalize the chalcogenide perovskite dataset
- Generate SISSO-derived primary features from ionic radii, electronegativities, and oxidation states
- Train a decision tree classifier on tolerance factor features
- Evaluate the SISSO tolerance factor (τ\*) against the Goldschmidt and Bartel tolerance factors
- Apply Platt scaling for probabilistic predictions
- Screen all possible ABX₃ compositions for structural stability

!!! info "SISSO features are pre-cached"
    The derived features are stored in `data/interim/features_sisso.csv` so you can skip the SISSO derivation step if you don't have `sissopp` installed.

---

### Step 2 — CrystaLLM Structure Generation & Classification

[`2_CrystaLLM_analysis.ipynb`](https://github.com/GarzonDiegoFEUP/chalcogenide-perovskite-screening/blob/main/notebooks/2_CrystaLLM_analysis.ipynb)

- Parse CrystaLLM-generated CIF files for candidate compositions
- Classify generated structures as corner-sharing vs edge-sharing octahedral networks
- Filter for topologically valid ABX₃ perovskite geometries
- Assess structural diversity across generated candidates

---

### Step 3 — Experimental Plausibility (GCNN)

[`3_Experimental_likelihood.ipynb`](https://github.com/GarzonDiegoFEUP/chalcogenide-perovskite-screening/blob/main/notebooks/3_Experimental_likelihood.ipynb)

- Assess crystal-likeness (synthesizability) using a pre-trained GCNN model
- Generate synthesizability scores for all candidate structures
- Rank candidates by experimental plausibility

---

### Step 4 — Bandgap Prediction (CrabNet)

[`4_bandgap_prediction.ipynb`](https://github.com/GarzonDiegoFEUP/chalcogenide-perovskite-screening/blob/main/notebooks/4_bandgap_prediction.ipynb)

- Train (or load) a CrabNet composition-based bandgap predictor
- Evaluate model accuracy on experimental halide perovskite and chalcogenide data
- Predict bandgaps for all surviving candidate compositions
- Filter candidates within the photovoltaic-relevant bandgap window

#### Step 4.1 — Encoder Comparison

[`4_1_encoder_comparison.ipynb`](https://github.com/GarzonDiegoFEUP/chalcogenide-perovskite-screening/blob/main/notebooks/4_1_encoder_comparison.ipynb)

- Compare different elemental encoding strategies for CrabNet
- Benchmark Pettifor-based vs default encoders

#### Step 4.2 — Training Data Size Ablation

[`4_2_data_size.ipynb`](https://github.com/GarzonDiegoFEUP/chalcogenide-perovskite-screening/blob/main/notebooks/4_2_data_size.ipynb)

- Analyse the effect of training set size on CrabNet bandgap prediction accuracy
- Determine the minimum data requirement for reliable composition-based predictions

---

### Step 5 — Sustainability Analysis

[`5_HHI_calculation.ipynb`](https://github.com/GarzonDiegoFEUP/chalcogenide-perovskite-screening/blob/main/notebooks/5_HHI_calculation.ipynb)

- Calculate the Herfindahl–Hirschman Index (HHI) for element supply concentration
- Integrate ESG (Environmental, Social, Governance) risk scores
- Combine with supply chain risk and earth-abundance metrics
- Produce the final multi-objective sustainability ranking

## Data Flow

| Step | Inputs | Outputs |
|------|--------|---------|
| 1 | `data/raw/` ionic radii, electronegativities | `data/interim/features_sisso.csv`, screened compositions |
| 2 | CrystaLLM CIF files in `data/crystaLLM/` | Classified structures, topology labels |
| 3 | Candidate CIF structures | Crystal-likeness scores |
| 4 | Candidate compositions | Predicted bandgaps |
| 5 | All previous outputs, `data/sustainability_data/` | Final ranked candidate list |
