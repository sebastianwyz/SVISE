# Analysis of the Frobenius Metric on Neural Networks

## Overview
This project investigates the variation of the **Frobenius Metric** across different types of neural networks (SBM, ER, RGR, BA) based on threshold \( t \) values and coupling constants (0.3, 0.5, 0.8, 1). The goal is to verify the effectiveness of the **SVISE method** on networks with a constant number of connections but varying topologies.

## Data and Method
Data was obtained by comparing estimated coefficient matrices and original adjacency matrices (scaled by the coupling constant). The **Frobenius Metric** was calculated by applying various thresholds to the estimated matrix and computing the normalized Frobenius norm of the difference between the thresholded estimated matrix and the modified adjacency matrix.

### Coupling Constants Analyzed
The coupling constants explored in this analysis are:
- **0.3**
- **0.5**
- **0.8**
- **1**

## Results

### Frobenius Metric Graphs
Below are the graphs of the Frobenius Metric for each network type and coupling constant.

#### 1. BA Network
![BA - C=0.3](Grafici_Frobenius/Frobenius_BA_C03.png)
![BA - C=0.5](Grafici_Frobenius/Frobenius_BA_C05.png)
![BA - C=0.8](Grafici_Frobenius/Frobenius_BA_C08.png)
![BA - C=1](Grafici_Frobenius/Frobenius_BA_C1.png)

#### 2. ER Network
![ER - C=0.3](Grafici_Frobenius/Frobenius_ER_C03.png)
![ER - C=0.5](Grafici_Frobenius/Frobenius_ER_C05.png)
![ER - C=0.8](Grafici_Frobenius/Frobenius_ER_C08.png)
![ER - C=1](Grafici_Frobenius/Frobenius_ER_C1.png)

#### 3. RGR Network
![RGR - C=0.3](Grafici_Frobenius/Frobenius_RGR_C03.png)
![RGR - C=0.5](Grafici_Frobenius/Frobenius_RGR_C05.png)
![RGR - C=0.8](Grafici_Frobenius/Frobenius_RGR_C08.png)
![RGR - C=1](Grafici_Frobenius/Frobenius_RGR_C1.png)

#### 4. SBM Network
![SBM - C=0.3](Grafici_Frobenius/Frobenius_SBM_C03.png)
![SBM - C=0.5](Grafici_Frobenius/Frobenius_SBM_C05.png)
![SBM - C=0.8](Grafici_Frobenius/Frobenius_SBM_C08.png)
![SBM - C=1](Grafici_Frobenius/Frobenius_SBM_C1.png)

## Discussion
From the graphs, we observe that:
- **[Observation 1]**: For instance, the SBM networks demonstrate a more pronounced variation in the Frobenius Metric with changes in the coupling constant.
- **[Observation 2]**: ER networks tend to maintain a stable Frobenius Metric up to a specific threshold \( t \), regardless of the coupling constant.

## Conclusion
The results indicate that the effect of the coupling constant on the Frobenius norm varies significantly depending on the network topology. This has implications for interpreting network connectivity, particularly concerning thresholding parameters in network analysis using the SVISE method.

## Notes
- Graphs were generated automatically and saved in the `Grafici_Frobenius` folder.
- Each graph shows the **Frobenius Metric** as a function of threshold \( t \), limited to the relevant range \( t \leq 1.5 \).

