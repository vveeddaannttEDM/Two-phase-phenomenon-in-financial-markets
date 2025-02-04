# Two-phase-phenomenon-in-financial-markets
# Two-Phase Phenomena in Financial Markets

## Overview
This project replicates the analysis from the paper *"Note on Two Phase Phenomena in Financial Markets"* by Shi-Mei Jiang et al. The study focuses on detecting the bifurcation phenomenon in financial markets, particularly in the Hang-Seng Index, where the probability distribution of returns transitions from unimodal (single peak) to bimodal (two peaks).

## Key Concepts
- **Bifurcation Phenomenon**: Transition in conditional probability distribution from unimodal to bimodal, indicating shifts between equilibrium and out-of-equilibrium market states.
- **Absolute Increment (I)**: Defined as \( I(t) = |y(t+1) - y(t)| \), where \( y(t) \) is the financial index at time \( t \).
- **Power-Law Behavior**: The absolute increment follows a power-law distribution characterized by an exponent \( \zeta_I \). Bifurcation appears when \( 1 < \zeta_I < 2 \).

## Features
1. Load and preprocess financial time series data.
2. Calculate absolute increments and fit a power-law model.
3. Detect bifurcation phenomenon by analyzing the conditional probability distribution of returns.
4. Compare empirical results with simulated time series data.

## Installation
Clone the repository:

```bash
git clone https://github.com/yourusername/two_phase_financial_market.git
cd two_phase_financial_market
```

Install required packages:

```bash
pip install -r requirements.txt
```

## Usage
Run the main script to analyze provided financial data or use the synthetic dataset:

```bash
python main.py
```

### Arguments:
- `--data_path`: (Optional) Path to the CSV file containing time series data. If not provided, synthetic data will be used.

## Data Format
Ensure your CSV file is formatted as follows:

| Timestamp       | Index Value |
|-----------------|-------------|
| 1994-07-01 10:00| 10000       |
| 1994-07-01 10:01| 10002       |
| ...             | ...         |

## Results
The script will output:
1. The power-law exponent \( \zeta_I \) of the absolute increment.
2. Visualization of the bifurcation phenomenon, showing transitions in the distribution of returns.
3. Comparison between real market data and simulated time series.

## Contributing
Feel free to fork this project and submit pull requests. For major changes, open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License.

## References
- Jiang, S.M., Cai, S.M., Zhou, T., Zhou, P.L. *Note on Two Phase Phenomena in Financial Markets.* [arXiv:0801.0108](http://arxiv.org/abs/0801.0108)
- Clauset, A., Shlizi, C.R., Newman, M.E.J. *Power-law Distributions in Empirical Data.* [arXiv:0706.1062](http://arxiv.org/abs/0706.1062)

