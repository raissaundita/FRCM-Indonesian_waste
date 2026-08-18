# A comparison of fuzzy clustering approach for provincial-level waste data in Indonesia

## Research Overview

This study compares the performance of four clustering methods for analyzing provincial-level waste characteristics in Indonesia:

* K-Means
* Fuzzy C-Means (FCM)
* Fuzzy Rough C-Means (FRCM)
* Firefly–Fuzzy Rough C-Means (Firefly-FRCM)

The dataset was obtained from the National Waste Management Information System (Sistem Informasi Pengelolaan Sampah Nasional, SIPSN) for the 2019–2025 period. The original data were collected at the regency/city level and include waste generation, waste composition, and waste source information. These data were subsequently preprocessed and aggregated to the provincial level.

The study focuses on comparing hard clustering, fuzzy clustering, fuzzy-rough clustering, and metaheuristic-assisted fuzzy-rough clustering approaches in identifying patterns of waste characteristics across Indonesian provinces.

Cluster quality was evaluated using the Silhouette Index, Dunn Index, Symmetric Purity, and Xie-Beni Index. The Silhouette Index, Dunn Index, and Symmetric Purity were used to compare all clustering methods, while the Xie-Beni Index was specifically applied to the fuzzy-based methods: FCM, FRCM, and Firefly-FRCM.

Statistical differences in clustering performance were further analyzed using the Friedman Test and Nemenyi post-hoc Test.

## Research Workflow

<p align="center">
  <img src="images/flowchart.png" width="500">
</p>

<p align="center">
  Research workflow of the final project.
</p>

## Research Stages

1. Data collection and understanding of the SIPSN dataset (2019–2025)
2. Data preprocessing
   - Data format and completeness checking
   - Data cleaning
   - Normality testing
   - Missing value handling
   - Regency/city-level data aggregation to the provincial level
   - Min-Max normalization
3. Parameter tuning for FRCM and Firefly-FRCM
4. Determination of the optimal number of clusters using the Elbow Method
5. Application of the clustering algorithms
6. Clustering performance evaluation
7. Statistical analysis using the Friedman and Nemenyi Tests
8. Cluster visualization and interpretation using t-SNE and spatial maps

Each clustering method was executed for 100 runs using different random initializations to obtain a more representative performance distribution and reduce the influence of a single initialization.

## Repository Structure

* Dataset/         -> Raw and aggregated datasets
* PREPROCE/        -> Data preprocessing
* TUNING/          -> Parameter tuning for FRCM and Firefly-FRCM
* ELBOW/           -> Optimal cluster number determination
* KLASTER_DATA/    -> Clustering results
* EVALUASI/        -> Silhouette, Dunn, Symmetric Purity, and Xie-Beni evaluation
* UJI STATIS/      -> Friedman and Nemenyi statistical tests
* VISUALISASI/     -> t-SNE and cluster map visualizations
* Dashboard/       -> Interactive analysis dashboard

## Tools and Libraries
* Python
* Pandas
* NumPy
* SciPy
* Scikit-Learn
* Optuna
* Matplotlib
* GeoPandas
* QGIS

## Author
Raissa Undita Estiningtyas <br>
Department of Mathematics <br>
Faculty of Science and Data Analytics <br>
Institut Teknologi Sepuluh Nopember (ITS) <br>
Surabaya, Indonesia
