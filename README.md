# Analysis of the Frobenius Metric on Neural Networks

## Overview
This project investigates the variation of the **Frobenius Metric** across different types of neural networks (SBM, ER, RGR, BA) based on threshold t values and different coupling constants (0.3, 0.5, 0.8, 1). The goal is to verify the effectiveness of the **SVISE method** on networks with a constant number of connections but different topologies.

### Frobenius Metric Graphs
Below are the graphs of the Frobenius Metric for each coupling constant K. Each graph includes all network types (SBM, ER, RGR, BA).

#### 1. Coupling Constant \( K = 0.3 \)
<img src="Graphs/Frobenius_C0.3.png" alt="K=0.3" width="600"/>

#### 2. Coupling Constant \( K = 0.5 \)
<img src="Graphs/Frobenius_C0.5.png" alt="K=0.5" width="600"/>

#### 3. Coupling Constant \( K = 0.8 \)
<img src="Graphs/Frobenius_C0.8.png" alt="K=0.8" width="600"/>

#### 4. Coupling Constant \( K = 1 \)
<img src="Graphs/Frobenius_C1.png" alt="K=1" width="600"/>

## Discussion
From the graphs, we observe that:
- **[Observation 1]**: The Frobenius Metric is notably higher for the Random Geometric Graph topology, while the other three topologies do not show a consistent advantage over one another.
- **[Observation 2]**: There is no clear preference for the value K, the best results, meaning those with lower Frobenius metrics at t=0, where K= 0.5 and 0.8.

## Conclusion
The results indicate that the effect of the coupling constant could cause the Frobenius norm to vary but there is no clear evidence for a preference towards a topology among SBM, BA or ER while the RGR performed worse than the other 3.

## Notes
- Graphs were generated automatically and saved in the `Graphs` folder.
- Each graph shows the **Frobenius Metric** as a function of threshold \( t \), limited to the relevant range \( t \leq 1.5 \).
