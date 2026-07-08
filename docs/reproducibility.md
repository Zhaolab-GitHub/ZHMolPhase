Detailed implementation and training settings of ZHMolPhase.

Protein language model: ESM-2 (esm2_t33_650M_UR50D)

ESM-2 scale: 650M parameters, 33 layers

ESM-2 embedding dimension: 1280

ESM-2 training status: Frozen

Structural input: PDB / AlphaFold2 structures

Residue coordinate: Cα coordinate; fallback to N then C if Cα is missing

Graph construction: Radius graph

Edge cutoff: 10 Å

Edge direction: Undirected, both i→j and j→i included

Edge feature: Euclidean distance in Å

RBF expansion: 24 basis functions, centers from 0 to 20 Å

Sequence encoder: 8 Mamba-convolution residual blocks

Hidden dimension: 256

Mamba parameters: n_ssm = 4, dt_rank = 1

Convolution kernels: 3, 5, 7

EGNN module: 7 EGNN layers

EGNN hidden dimension: 192

Batch size: 4

Maximum epochs: 30

Early stopping: Validation AUC, patience = 4

LR scheduler: ReduceLROnPlateau, validation AUC, factor = 0.5, patience = 1, min LR = 1 × 10⁻⁶

Loss function: BCEWithLogitsLoss

Cross-validation: 5-fold stratified cross-validation

Final prediction: Mean probability ensemble of 5 fold models

Maximum sequence length: Less than 5000 in training

Long-protein processing: No truncation or block segmentation

Padding: Dynamic padding within each batch

Classification threshold: 0.5 for threshold-dependent metrics

Primary metric: ROCAUC

Dropout: 0.15

Optimizer: AdamW

Learning rate: 2 × 10⁻⁴

Weight decay: 1 × 10⁻²
