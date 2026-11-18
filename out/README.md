
# Lifespan Extension of AI Accelerators

### Balancing Embodied and Operational Emissions through Workload Scheduling

A simulation-based research project investigating how GPU replacement cycles, workload scheduling, and power management influence lifecycle carbon emissions and electronic waste.

## Overview

This project examines the environmental trade-offs between retaining older AI accelerators and replacing them with newer, more energy-efficient GPUs.

The study combines:

- Carbon break-even analysis for GPU upgrades.
- Fleet-level hardware replacement and e-waste modeling.
- Discrete-event workload scheduling simulation.
- Sensitivity and uncertainty analysis.

## Hardware Configurations

The simulation considers three NVIDIA GPU generations:

| GPU | Peak FP16 Tensor Performance | Board TDP |
|---|---|---|
| V100 | 125 TFLOPS | 300 W |
| A100 | 312 TFLOPS | 400 W |
| H100 | 989 TFLOPS | 700 W |

These values are peak specification proxies, not measured application throughput.

## Experiments

| ID | Description |
|---|---|
| E1 | GPU replacement carbon break-even analysis |
| E2 | Fleet refresh cycles and e-waste generation |
| E3 | Comparison of five workload scheduling policies |
| E4 | Condition-based retirement (future work) |
| E5 | Sensitivity and Monte Carlo uncertainty analysis |

## Requirements

- Python 3.10 or newer
- NumPy
- Matplotlib

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Running the Simulation

From the project directory:

```bash
python ewaste_model.py
```

The program uses a fixed random seed of 42 for reproducibility.

## Generated Outputs

The script generates:

```text
out/
├── results.json
├── fig_breakeven.pdf
├── fig_ewaste.pdf
└── fig_tornado.pdf
```

The JSON file contains experiment results, while the PDF files contain figures used in the research paper.

## Main Findings

Under the central simulation assumptions:

- V100 to A100 carbon break-even: approximately 2.11 years.
- A100 to H100 carbon break-even: approximately 1.30 years.
- V100 to H100 carbon break-even: approximately 0.44 years.
- Extending the modeled refresh cycle from three to five years reduces cumulative retired hardware mass by approximately 24%.
- Efficiency-first scheduling performs better than simple age-tiered scheduling in the evaluated scenarios.

These findings depend on the assumptions used in the simulation.

## Important Limitations

- GPU aging and wear are modeled using assumed parameters, not validated physical degradation measurements.
- The scheduling experiment estimates retirement-equivalent wear rather than actual GPU failures or physical disposal.
- Embodied carbon is estimated using secondary sources and simplified allocation methods.
- GPU performance is approximated using theoretical peak throughput ratios.
- Future GPU generations in the fleet model are hypothetical.
- The results should not be interpreted as experimentally verified GPU lifespan extensions.

## Reproducibility

To reproduce the results:

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install the dependencies.
4. Run `python ewaste_model.py`.
5. Compare the generated `out/results.json` with the reported values.

## Authors

- Bishal Chaudhary
- Dikshyant Lamsal
- Jenish Karki
- Kirtan Shrestha
- Ayan Pokhrel

Department of Artificial Intelligence and Machine Learning  
CMR Institute of Technology, Bengaluru, India

The remaining co-author details will be updated when confirmed.

## Research Status

Simulation-based proof of concept. Further validation using real GPU telemetry and measured workload performance is required.
