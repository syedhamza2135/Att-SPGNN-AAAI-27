# AttGNN-SP: Attention-Based Graph Neural Networks for High-Fidelity Synthetic Population Synthesis

A Graph Neural Network (GNN) based approach for synthetic population generation. This research uses Graph Attention Networks (GAT) to generate realistic synthetic individuals and households that match census-derived statistical distributions.

## Features

- **Individual Generation**: Generate synthetic persons with 7 demographic attributes (age, sex, ethnicity, religion, marital status, qualification, household composition)
- **Household Generation**: Generate synthetic households with 6 attributes (composition, ethnicity, religion, tenure, size, rooms)
- **Household Assignment**: Optimally assign individuals to households respecting multiple constraints
- **Multi-task Learning**: Jointly learn multiple attribute distributions for consistency
- **Interactive Visualizations**: Plotly-based HTML plots comparing actual vs predicted distributions
- **Batch Processing**: Process multiple geographic areas in sequence

## Requirements

- Python 3.10 or higher
- PyTorch 2.0+ (with GPU support recommended)
- CUDA 11.8+ (for NVIDIA GPUs) or MPS (for Apple Silicon)

## Project Structure

```
ATT-SPGNN-AAAI-27/
├── code/
│   ├── generateIndividuals.py       # Individual/person generation using GAT
│   ├── generateHouseholds.py        # Household generation using GAT
│   ├── assignHouseholds.py          # Person-to-household assignment using bipartite GAT
│   ├── main.py                      # Interactive menu interface
│   ├── evaluation.py                # Model evaluation utilities
│   ├── utils/                       # Utility scripts
│   │   ├── createGlossary.py        # Create master glossary from crosstables
│   │   ├── plotConvergencePerformance.py  # Generate convergence/performance plots
│   │   ├── runAssignmentHPTuning.py       # Batch hyperparameter tuning
│   │   └── runMultipleAreas.py            # Batch processing for multiple areas
│   └── outputs/                     # Generated outputs (created automatically)
├── data/
│   ├── raw-data/                    # Original census data
│   ├── preprocessed-data/           # Processed crosstables and marginal tables
│   │   ├── individuals/             # Person-level marginal distributions
│   │   └── crosstables/             # Joint distribution tables
│   └── geodata/                     # Geographic boundary files (MSOA shapefiles)
├── requirements.txt                 # Python dependencies
└── README.md
```

## Installation

### Step 1: Clone and Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd SPH_GNN

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 2: Install PyTorch (Platform-Specific)

Choose the appropriate installation based on your system:

#### 🍎 macOS with Apple Silicon (M1/M2/M3)

```bash
# PyTorch with MPS (Metal Performance Shaders) support
pip install torch torchvision torchaudio
```

The code automatically detects and uses MPS acceleration. You'll see `Using device: mps` when running.

#### 🪟 Windows with NVIDIA GPU

```bash
# PyTorch with CUDA support (check your CUDA version first)
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 12.4:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

To check your CUDA version: `nvidia-smi` (look for "CUDA Version" in the output)

#### 🐧 Linux with NVIDIA GPU

```bash
# PyTorch with CUDA support
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 12.4:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

#### 💻 CPU Only (Any Platform)

```bash
# CPU-only installation (slower, but works everywhere)
pip install torch torchvision torchaudio
```

### Step 3: Install Other Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
# Check PyTorch and GPU availability
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'MPS: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')"
```

Expected output (varies by platform):
- **NVIDIA GPU**: `CUDA: True`
- **Apple Silicon**: `MPS: True`
- **CPU only**: Both `False`

## Usage

### Interactive Menu (Recommended)

```bash
cd code
python main.py
```

The menu provides access to:
1. **Individual Generation** - Generate synthetic persons for a specific area
2. **Household Generation** - Generate synthetic households for a specific area
3. **Household Assignment** - Assign persons to households with constraint optimization
4. **Create Master Glossary** - Combine crosstable glossaries
5. **Convergence & Performance Plots** - Analyze training results
6. **Batch Processing** - Run multiple areas in sequence
7. **Model Evaluation** - Compare outputs against census data

### Command Line Execution

```bash
# Generate individuals for a specific area
python code/generateIndividuals.py --area_code E02005940

# Generate households for the same area
python code/generateHouseholds.py --area_code E02005940

# Assign persons to households
python code/assignHouseholds.py --area_code E02005940
```

### Available Area Codes

The project includes data for 17 Oxford MSOA areas:
```
E02005940, E02005941, E02005942, E02005943, E02005944,
E02005945, E02005946, E02005947, E02005948, E02005949,
E02005950, E02005951, E02005953, E02005954, E02005955,
E02005956, E02005957
```

## Output Files

After running the generation scripts, outputs are saved in `code/outputs/`:

```
outputs/
├── individuals_{area_code}/
│   ├── person_nodes.pt                    # Generated person tensor [N, 7]
│   ├── generated_individuals.csv          # Human-readable person table
│   ├── crosstable_comparison.html         # Interactive comparison plots
│   └── convergence_data.csv               # Training metrics
├── households_{area_code}/
│   ├── household_nodes.pt                 # Generated household tensor [M, 6]
│   ├── generated_households.csv           # Human-readable household table
│   ├── household_crosstable_comparison.html
│   └── convergence_data.csv
└── assignment_hp_tuning_{area_code}/
    ├── final_assignments.pt               # Person-to-household assignments
    ├── hp_tuning_results.csv              # Hyperparameter search results
    └── accuracy_over_epochs.png           # Training progress plot
```

## GPU Memory Requirements

| Component | Recommended GPU Memory |
|-----------|----------------------|
| Individual Generation | 4-8 GB |
| Household Generation | 4-8 GB |
| Household Assignment | 8-16 GB |

For systems with limited GPU memory:
- Reduce `hidden_channel_options` in the scripts (e.g., use `[128]` instead of `[128, 256]`)
- The code includes automatic memory management and early stopping

## Troubleshooting

### MPS (Apple Silicon) Issues

If you encounter `NotImplementedError` on MPS:
```python
# Some operations may fall back to CPU automatically
# The code handles this gracefully
```

### CUDA Out of Memory

If you get OOM errors:
1. Reduce `hidden_channel_options` to `[128]`
2. Increase `edge_dropout` to `0.2`
3. Reduce `num_epochs`

### Missing Dependencies

```bash
# Reinstall all dependencies
pip install --upgrade -r requirements.txt
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{attgnn-sp,
  title = {AttGNN-SP: Attention-Based Graph Neural Networks for High-Fidelity Synthetic Population Synthesis},
  author = {Shah, Syed Saad Ullah},
  year = {2026},
  url = {https://github.com/SaadShah11/AttGNN-SP}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Syed Saad Ullah Shah
