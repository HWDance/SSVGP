# SSVGP: Spike and Slab Variational Gaussian Process

This repository provides a Python implementation of the Spike and Slab Variational Gaussian Process (SSVGP) algorithm, as introduced in the paper:  
**Fast and Scalable Spike and Slab Variable Selection in High-Dimensional Gaussian Processes**  
*Hugh W. Dance and Brooks Paige (2022)*  
[arXiv:2111.04558](https://arxiv.org/abs/2111.04558)

## Overview

SSVGP is a Bayesian variable selection method for Gaussian Processes (GPs) that incorporates spike-and-slab priors over inverse lengthscale parameters. This approach enables effective identification of relevant variables in high-dimensional settings while maintaining computational efficiency.

Key features:

- Employs a-CAVI (amortized Coordinate Ascent Variational Inference) for scalable inference.
- Utilizes Bayesian Model Averaging (BMA) over spike precisions to enhance robustness.
- Incorporates dropout pruning to enforce sparsity in the model.
- Demonstrates competitive performance compared to MCMC-based spike-and-slab GPs, with significantly reduced computational cost.

## Repository Contents

- `SSVGP.py`: Core implementation of the SSVGP algorithm.
- `SSVGP demo.ipynb`: Jupyter notebook demonstrating the application of SSVGP on a synthetic dataset.
- `AISTATS_poster.png` and `AISTATS_presentation.pdf`: Poster and presentation slides from the AISTATS 2022 conference.
- `LICENSE`: Apache 2.0 License.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/HWDance/SSVGP.git
   cd SSVGP

2. **Install required dependencies:**
   ```python
   pip install numpy scipy matplotlib jupyter
   ```

## Getting Started
To familiarize yourself with the SSVGP implementation:


1. **Launch Jupyter Notebook:**

   ```bash
   jupyter notebook

2. **Run the Demo**
   ```bash
   SSVGP demo.ipynb.

This notebook demonstrates the application of SSVGP on a synthetic dataset, showcasing its variable selection capabilities.

## Using SSVGP in Your Projects
To apply SSVGP to your own datasets, import the SSVGP module:

```python
from SSVGP import SSVGP
```
and follow the workflow in `SSVGP demo.ipynb`.

## Citation
If you use this code or the SSVGP method in your research, please cite the following paper:

@inproceedings{dance2022fast,
  title={Fast and Scalable Spike and Slab Variable Selection in High-Dimensional Gaussian Processes},
  author={Dance, Hugh W and Paige, Brooks},
  booktitle={Proceedings of the 25th International Conference on Artificial Intelligence and Statistics},
  year={2022},
  url={https://arxiv.org/abs/2111.04558}
}

## License
This project is licensed under the Apache 2.0 License. See the LICENSE file for details.

Contact
For questions or feedback, please contact:

Hugh W. Dance
PhD Researcher, Machine Learning
Gatsby Computational Neuroscience Unit, UCL
uctphwd@ucl.ac.uk


