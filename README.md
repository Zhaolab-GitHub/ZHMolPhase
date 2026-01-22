# [ZHMolPhase](http://zhaoserver.com.cn/ZHMolPhase/ZHMolPhase.html)

## Overview of ZHMolPhase: 
**ZHMolPhase is a computational framework for the prediction and interpretation of phase-separating proteins and their driving regions.** 

**We also provide an online calculation server for convenient use:** 
[Click to access the online server](http://zhaoserver.com.cn/ZHMolPhase/ZHMolPhase.html)

## Installation and Environment Setup

### Prerequisites:
* OS: Ubuntu 20.04 LTS
* Python ≥ 3.8 (tested on 3.8.12)
* NVIDIA driver with CUDA support (tested on NVIDIA GeForce 2080)

### Installation Steps:
Follow the steps below to set up the environment. The installation has been tested in a clean environment.  

### Option 1: Recommended (conda environment file)
```
conda env create -f environment.yml
conda activate ZHMolPhase_env
```

### Option 2: Manual installation
```
conda create -n ZHMolPhase_env python=3.8.12 -c conda-forge
conda activate ZHMolPhase_env
```
```
pip install numpy==1.24.4
pip install pandas==2.0.3
pip install tqdm==4.67.1
pip install biopython==1.83
pip install einops==0.8.1
pip install torch==2.4.1
```

## Usage

Example input files are provided in the `example/` directory, including:
- FASTA files containing protein sequences
- PDB files containing protein structures
- Precomputed protein language model embeddings (LLM)

The LLM features are generated using **esm2_t33_650M_UR50D**
(https://huggingface.co/facebook/esm2_t33_650M_UR50D).

### Download trained model

Download the pre-trained model weights from Zenodo:
https://zenodo.org/uploads/18335181

After downloading, extract the archive and place the model files in the same directory as specified by the `--ckpt_dir` argument.


### Predict Phase-Separating Proteins
```
python predict.py --names_txt name.txt --fasta_dir example/sequence --pdb_dir example/pdb --llm_dir example/LLM --ckpt_dir ckpts --out example/output/score.txt
```
### Predict key residues

#### Step 1: compute residue-level occlusion scores
```
python residue_score.py --names_txt name.txt --fasta_dir example/sequence --pdb_dir example/pdb --llm_dir example/LLM --ckpt_dir ckpts --window 11 --out_dir example/output/score
```
#### Step 2: extract predicted key residues
```
python get_key_region.py example/output/score/<protein_id>_occlusion.tsv example/output/score/<protein_id>.txt
```
where `<protein_id>` corresponds to the protein name specified in `name.txt`.

## Datasets
Training and test datasets are provided in the `dataset/` directory.

## Contact
For questions or suggestions regarding ZHMolPhase, please contact:


Yunjie Zhao; E-mail: yjzhaowh@ccnu.edu.cn  
Chengwei Zeng; E-mail: cwzengwuhan@mails.ccnu.edu.cn
