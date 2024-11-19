# Exploring Mental Health in Tech Jobs: Uncovering Key Patterns Through Clustering
## Project Overview
The mental health of employees has become a critical topic, especially in technology-oriented workplaces where cognitive workload and stress levels are often elevated. This project examines the mental health of employees using data-driven techniques, with the goal of providing actionable insights for Human Resources (HR) departments to design pre-emptive mental health programs.

This case study uses the **OSMI Mental Health in Tech** dataset (2016) from Kaggle to explore attitudes toward mental health in the tech industry and the prevalence of mental health disorders among tech professionals. The analysis employs unsupervised machine learning techniques, such as clustering, to group respondents based on distinct mental health-related characteristics.

## Motivation
Mental health challenges have significant individual, societal, and economic impacts. Key findings from existing research underscore the urgency:

- A sharp increase in days of incapacity for work due to mental illness in Germany since 1997 (Statista, 2024).
- The World Health Organization (WHO) links mental health issues to productivity losses and workplace stress (WHO, 2024).
- Studies, such as McGrath et al. (2023), project that half of the world’s population will experience mental health disorders at some point in their lives.

This project aims to provide HR teams with tailored, actionable recommendations to address mental health issues proactively, reducing their impact on employees and organizations.

## Data
The analysis is based on the **OSMI Mental Health in Tech** dataset from Kaggle, which contains over 1,400 survey responses. The dataset can be accessed via the following link: https://www.kaggle.com/datasets/osmi/mental-health-in-tech-2016        
The dataset assesses attitudes toward mental health in the tech workplace and information on mental health disorders among tech professionals.


## Challenges
High dimensionality: Many features in the dataset.
- Missing values: Incomplete survey responses.
- Non-standardized textual inputs: Free-text responses require preprocessing.

## Methodology
This case study involves the following steps:

1. Data Preprocessing and Feature Engineering
    - Cleaning and preparing data for analysis.
    - Handling missing values and non-standardized textual inputs.
    - Engineering new features from textual inputs to enhance interpretability.

2. Feature Selection
    - Identifying important variables for analysis to reduce complexity while retaining key information.

3. Dimensionality Reduction
   - Applying techniques like Multi-dimensional Scaling (MDS) to simplify and prepare high-dimensional data for analysis.

4. Clustering
    - Using unsupervised machine learning algorithms to identify patterns in the data.
    - Grouping respondents into clusters based on shared mental health-related characteristics.
    - Analyzing distribution of features in the clusters using common descriptive statistics.
    - Visualize findings.

5. Interpretation and Recommendations
    - Analyzing clusters to provide HR with actionable insights.
    - Recommending tailored mental health interventions for employees based on cluster characteristics.

## Tools and Techniques
- **Python Libraries**: pandas, numpy, sklearn, matplotlib, seaborn.
- **Machine Learning Techniques**: Clustering (i.e., k-means clustering), dimensionality reduction (i.e., MDS).
- **Data Source**: OSMI Mental Health in Tech Dataset (Kaggle, 2024).


## How to Use This Repository
Download data from https://www.kaggle.com/datasets/osmi/mental-health-in-tech-2016      
Install required Python libraries:

```pip install -r requirements.txt```

Run the file ```Dataprep.py``` for the step of Data Preprocessing and Feature Engineering. Run the ```Analysis.py``` file for Feature Selection, Dimensionality Reduction, and Cluster Analysis.

## References
- Kaggle (2024). OSMI Mental Health in Tech Survey 2016. https://www.kaggle.com/datasets/osmi/mental-health-in-tech-2016 [Accessed 31.10.24].
- McGrath, J. J., Al-Hamzawi, A., Alonso, J., Altwaijri, Y., Andrade, L. H., Bromet, E. J., Bruffaerts, R., Caldas de Almeida, J. M., Chardoul, S., Chiu, W. T., Degenhardt, L., Demler, O. V., Ferry, F., Gureje, O., Haro, J. M., Karam, E. G., Karam, G., Khaled, S. M., Kovess-Masfety, V., Magno, M., Medina-Mora, M. E., Moskalewicz, J., Navarro-Mateu, F., Nishi, D., Plana-Ripoll, O., Posada-Villa, J., Rapsey, C., Sampson, N. A., Stagnaro, J. C., Stein, D. J., ten Have, M., Torres, Y., Vladescu, C., & Woodruff, P. W. & Zaslavsky, A. M. (2023). Age of onset and cumulative risk of mental disorders: a cross-national analysis of population surveys from 29 countries. The Lancet Psychiatry, 10(9), 668-681.
- Robert Koch Institut (2024). Mental Health Surveillance Beobachtung der psychischen Gesundheit der erwachsenen Bevölkerung in Deutschland. https://public.data.rki.de/t/public/views/hf-MHS_Dashboard/Dashboard?%3Aembed=y&%3AisGuestRedirectFromVizportal=y [Accessed 28.10.24.].
- Statista (2024). Arbeitsunfähigkeitstage aufgrund psychischer Erkrankungen in Deutschland nach Geschlecht in den Jahren 1997 bis 2023. https://de.statista.com/statistik/daten/studie/254192/umfrage/entwicklung-der-au-tage-aufgrund-psychischer-erkrankungen-nach-geschlecht/  [Accessed 28.10.24.].
- World Health Organization (2024). Mental health Impact. https://www.who.int/health-topics/mental-health#tab=tab_2 [Accessed 28.10.24.].