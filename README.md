# Latent Geometry and Out-of-Distribution Transfer in LLM Deception Probes

## Key Takeaway
[Draft finding: Linear probes trained on factual falsehood generalized poorly to instructed lies (dropping from XX% to YY%), driven primarily by directional rotation rather than an activation offset shift; retraining after projecting out the falsehood direction confirmed that deceptive intent retains an independent, decodable signal.]

## Summary of Results
* **Directional Rotation vs. Threshold Offset:**
  * The cosine similarity between factual falsehood ($\vec{w}_{\text{falsehood}}$) and instructed deception ($\vec{w}_{\text{deception}}$) peaked at only [0.AA] at layer [L], well below the within-distribution split-half reliability ceiling of [0.BB].
  * Freezing the probe direction and refitting *only* the scalar threshold $b$ on the target distribution [did / did not] recover accuracy (shifting from [YY]% to [ZZ]%), indicating that generalization failure is driven by [directional rotation / an activation offset shift].
* **Decoupling Falsehood from Deception:**
  * Symmetrically prompting both classes in Distribution B (instructed truth vs. instructed lie) controlled for instruction tokens.
  * Projecting out the factual falsehood vector $\vec{v}_{\text{falsehood}}$ and retraining on the residual subspace yielded [WW]% accuracy, showing that the model's representation of lying [is / is not] merely an artifact of evaluating a false claim.
* **Instruction Confound Check:** Testing the deception probe on benign prompts containing conversational formatting confirmed it does not fire purely on instruction presence.

## Research Motivation
Anthropic's "Fine-Tuned Lie Detectors Failed to Generalize" found that deception detectors often fail to transfer across distributions. This project investigates the underlying latent geometry: when a probe fails to transfer, is it due to directional rotation, threshold shifts, or confounding between dishonesty and factual falsehood?

## Experimental Setup
* **Model:** `google/gemma-2-2b-it`
* **Data Splits:**
  * *Distribution A (Factual Veracity):* True vs. false statements from Marks & Tegmark's *Geometry of Truth*.
  * *Distribution B (Instructed Intent):* Symmetrically instructed pairs across identical facts:
    * Label 0: Prompted with *"Answer truthfully"* $\rightarrow$ True response.
    * Label 1: Prompted with *"Answer falsely"* $\rightarrow$ False response.
* **Probing Methods:** Difference-of-means vectors and $L_2$-regularized Logistic Regression across residual stream layers (layers 10 to 22).

## Results & Analysis

### 1. Cross-Distribution Transfer and Threshold Refit
| Layer | In-Dist A (Falsehood) | Zero-Shot Transfer to B | Threshold Refit on B | Split-Half Ceiling (B) |
| :--- | :--- | :--- | :--- | :--- |
| Layer 10 | [XX.X]% | [YY.Y]% | [ZZ.Z]% | [0.BB] |
| Layer 14 | [XX.X]% | [YY.Y]% | [ZZ.Z]% | [0.BB] |
| Layer 18 | [XX.X]% | [YY.Y]% | [ZZ.Z]% | [0.BB] |
| Layer 22 | [XX.X]% | [YY.Y]% | [ZZ.Z]% | [0.BB] |

*(Insert: `results/plots/transfer_and_recovery.png`)*

### 2. Layer-wise Directional Alignment
*(Insert plot showing layer-wise cosine similarity between probe directions compared to the split-half upper bound: `results/plots/cosine_vs_ceiling.png`)*

### 3. Orthogonalization Check
*(Insert table/plot showing probe performance on instructed lies after projecting out $\vec{v}_{\text{falsehood}}$ and retraining on the residual activation subspace)*

## Methodological Context
This geometric framework adapts empirical analysis methods from my Master's thesis, where representations achieved 88% linear SVM separability despite a negative silhouette score (-0.028) across DBSCAN clusters—demonstrating that target properties can align cleanly along linear directions even without forming isolated spatial clusters.

## Limitations
* Evaluated on a single 2B parameter model (`gemma-2-2b-it`); representations and geometric stability may behave differently at frontier model scale.
* Subspace orthogonalization was conducted via single-direction projection; more comprehensive concept erasure methods (such as LEACE) could be applied to eliminate multi-dimensional linear correlations.
* Evaluated on teacher-forced completions rather than multi-step agentic rollouts.

## Reproducibility
```bash
git clone [https://github.com/berserkersap/deception-probe-geometry.git](https://github.com/berserkersap/deception-probe-geometry.git)
cd deception-probe-geometry
pip install -r requirements.txt
python run_experiments.py --model google/gemma-2-2b-it --layers 10 14 18 22
