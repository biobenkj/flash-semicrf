# Disclosure of AI-assisted development

This project incorporates contributions produced with the assistance of AI-based
software development tools. These tools were used for ideation, code generation,
debugging, and documentation support. Final implementations were made by the
authors, and all code has undergone manual review and testing. The project
author(s) assumes full responsibility for the accuracy, integrity, and licensing
compliance of all included code.

## Algorithmic Techniques

The Triton kernel implementations use the following established techniques:

- **Online logsumexp**: The streaming logsumexp pattern (tracking running max and
  sum-of-exponentials) is used for numerically stable reductions in the backward
  kernel. This technique is well-documented in Flash Attention's Triton
  implementation by Tri Dao (Dao-AILab/flash-attention).

- **Loop tiling**: Register pressure management via tiled computation, as used in
  Mamba's SSM kernels (state-spaces/mamba).
