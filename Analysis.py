import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import KMeans
import seaborn as sns
from scipy.stats import spearmanr

df = pd.read_csv('C:/Users/sophi/Desktop/MentalHealthClean.csv')

#select columns from dataset and delete possible missing values
selected_columns = ['PreviousEmployers', 'ProfTreatmentMHIssue', 'Age', 'RemoteWork', 'NumberOfEmployees',
                    'RequestMedicalLeave',
                    'MHBenefitsPrevEmployers', 'DiscussionMHPrevEmployers', 'ResourcesMHPrevEmployers',
                    'ProtectionAnonymityPrevEmployers',
                    'NegConsequencesDiscussionMHPrevEmployer', 'NegConsequencesDiscussionPHPrevEmployer',
                    'DiscussionMHPrevCoworker',
                    'DiscussionMHPrevSupervisor', 'SeriousnessMHPrevEmployer', 'NegConsequencesCoworkersPrevEmployer',
                    'ShareMI',
                    'MHInterferenceWorkEffTreat', 'MHInterferenceWorkNoTreat',
                    'KnowledgeMHCEmployerCoverage_I am not sure',
                    'KnowledgeMHCEmployerCoverage_No', 'KnowledgeMHCEmployerCoverage_Yes',
                    "DiscussionMHEmployer_I don't know",
                    'DiscussionMHEmployer_No', 'DiscussionMHEmployer_Yes', "EmployerResourcesMH_I don't know",
                    'EmployerResourcesMH_No',
                    'EmployerResourcesMH_Yes', "AnonymityProtection_I don't know", 'AnonymityProtection_No',
                    'AnonymityProtection_Yes',
                    'DiscussionNegConsequencesMH_Maybe', 'DiscussionNegConsequencesMH_No',
                    'DiscussionNegConsequencesMH_Yes',
                    'DiscussionNegConsequencesPH_Maybe', 'DiscussionNegConsequencesPH_No',
                    'DiscussionNegConsequencesPH_Yes',
                    'DiscussionMHCoworkers_Maybe', 'DiscussionMHCoworkers_No', 'DiscussionMHCoworkers_Yes',
                    'DiscussionMHDirectSupervisor_Maybe',
                    'DiscussionMHDirectSupervisor_No', 'DiscussionMHDirectSupervisor_Yes',
                    "EmployerMHSerious_I don't know", 'EmployerMHSerious_No',
                    'EmployerMHSerious_Yes', 'NegConsequencesOpenMH_No', 'NegConsequencesOpenMH_Yes',
                    'AwarenessMHCarePrevEmployers_I was aware of some',
                    'AwarenessMHCarePrevEmployers_N/A (not currently aware)',
                    'AwarenessMHCarePrevEmployers_No, I only became aware later',
                    'AwarenessMHCarePrevEmployers_Yes, I was aware of all of them', 'PHIssueInterview_Maybe',
                    'PHIssueInterview_No',
                    'PHIssueInterview_Yes', 'MHIssuesInterview_Maybe', 'MHIssuesInterview_No', 'MHIssuesInterview_Yes',
                    'MHHurtCareer_Maybe',
                    "MHHurtCareer_No, I don't think it would", 'MHHurtCareer_No, it has not',
                    'MHHurtCareer_Yes, I think it would',
                    'MHHurtCareer_Yes, it has', 'NegViewedByTeam_Maybe', "NegViewedByTeam_No, I don't think they would",
                    'NegViewedByTeam_No, they do not', 'NegViewedByTeam_Yes, I think they would',
                    'NegViewedByTeam_Yes, they do',
                    'UnsupportiveResponseMH_Maybe/Not sure', 'UnsupportiveResponseMH_No',
                    'UnsupportiveResponseMH_Yes, I experienced',
                    'UnsupportiveResponseMH_Yes, I observed', 'NegatveImpactRevealMH_Maybe', 'NegatveImpactRevealMH_No',
                    'NegatveImpactRevealMH_Yes', "HistoryMI_I don't know", 'HistoryMI_No', 'HistoryMI_Yes',
                    'PastMI_Maybe',
                    'PastMI_No', 'PastMI_Yes', 'CurrentMHDisorder_Maybe', 'CurrentMHDisorder_No',
                    'CurrentMHDisorder_Yes',
                    'ProfessionalDiagnosis_No', 'ProfessionalDiagnosis_Yes', 'Gender_Diverse', 'Gender_Female',
                    'Gender_Male',
                    'ContinentLive_Asia', 'ContinentLive_Europe', 'ContinentLive_North America',
                    'ContinentLive_Oceania',
                    'ContinentLive_South America', 'ContinentWork_Asia', 'ContinentWork_Europe',
                    'ContinentWork_North America',
                    'ContinentWork_Oceania', 'ContinentWork_Other', 'ContinentWork_South America', 'LeadershipPosition',
                    'SupportPosition', 'MI_Anxiety Disorders', 'MI_Attention Deficit Disorders',
                    'MI_Autism Spectrum Disorders', 'MI_Eating Disorders', 'MI_Mood Disorders',
                    'MI_Obsessive-Compulsive and Related Disorders', 'MI_Personality Disorders',
                    'MI_Post-Traumatic and Stress-Related Disorders', 'MI_Psychotic Disorders',
                    'MI_Substance Use and Addictive Disorders']
df_selected = df[selected_columns].dropna()

#standardize data in preparation for MDS
df_scaled = pd.DataFrame(MinMaxScaler().fit_transform(df_selected), columns=df_selected.columns)

#feature selection based on scaled features and corresponding variances
feature_variances = df_scaled.var()
variance_df = feature_variances.reset_index()
variance_df.columns = ['Feature', 'Variance']
sorted_variance_df_scaled = variance_df.sort_values(by='Variance', ascending=False)
print(sorted_variance_df_scaled)
high_variance_features_scaled = feature_variances[feature_variances > 0.24].index.tolist()
print(high_variance_features_scaled)

#feature selection based on unscaled features and corresponding variances
feature_variances = df_selected.var()
variance_df = feature_variances.reset_index()
variance_df.columns = ['Feature', 'Variance']
sorted_variance_df_unscaled = variance_df.sort_values(by='Variance', ascending=False)
print(sorted_variance_df_unscaled)
high_variance_features_unscaled = feature_variances[feature_variances > 0.4].index.tolist()
print(high_variance_features_unscaled)

#Feature selection according to variance threshold of 0.24 and scaled data
threshold = 0.24
selector_scaled = VarianceThreshold(threshold=threshold)
selected_features_scaled = selector_scaled.fit_transform(df_scaled)
selected_feature_names_scaled = df_scaled.columns[selector_scaled.get_support()]
reduced_feature_df_scaled = pd.DataFrame(selected_features_scaled, columns=selected_feature_names_scaled)
print(reduced_feature_df_scaled)

#Feature selection according to variance threshold of 0.4 and unscaled data
threshold = 0.4
selector_unscaled = VarianceThreshold(threshold=threshold)
selected_features_unscaled = selector_unscaled.fit_transform(df_selected)
selected_feature_names_unscaled = df_selected.columns[selector_unscaled.get_support()]
reduced_feature_df_unscaled = pd.DataFrame(selected_features_unscaled, columns=selected_feature_names_unscaled)
print(reduced_feature_df_unscaled)

#Feature selection according to feasibility
selected_columns2 = ['PreviousEmployers', 'Age', 'RemoteWork', 'NumberOfEmployees', 'PastMI_Maybe',
                     'PastMI_No', 'PastMI_Yes', 'CurrentMHDisorder_Maybe', 'CurrentMHDisorder_No',
                     'CurrentMHDisorder_Yes',
                     'Gender_Diverse', 'Gender_Female', 'Gender_Male', 'ContinentWork_Asia', 'ContinentWork_Europe',
                     'ContinentWork_North America', 'ContinentWork_Oceania', 'ContinentWork_South America',
                     'LeadershipPosition',
                     'SupportPosition']
df_feasible = df[selected_columns2].dropna()

#MDS and cluster analysis for scaled data
mds = MDS(n_components=2, random_state=42)
mds_transformed = mds.fit_transform(reduced_feature_df_scaled)
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(mds_transformed)
mds_df = pd.DataFrame(mds_transformed, columns=['Dimension 1', 'Dimension 2'])
df_combined = pd.concat([reduced_feature_df_scaled, mds_df], axis=1)
df_combined['Cluster'] = clusters


inertia = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(mds_transformed)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

print(df_combined)
df_combined.to_csv('C:/Users/sophi/Desktop/ClusterAnalysisScaledData.csv', index=False)

plt.scatter(df_combined['Dimension 1'], df_combined['Dimension 2'], c=df_combined['Cluster'], cmap='viridis', s=100)
plt.xlabel('Dimension 1', fontsize=18)
plt.ylabel('Dimension 2', fontsize=18)
plt.colorbar(label='Cluster')
plt.show()

#Projection matrix (Heatmap) for scaled data
reduced_feature_df_scaled_array = np.array(reduced_feature_df_scaled)
correlation_matrix = pd.DataFrame({
    'Dimension1': [spearmanr(reduced_feature_df_scaled_array[:, i], mds_df['Dimension 1'])[0] for i in
                   range(reduced_feature_df_scaled_array.shape[1])],
    'Dimension2': [spearmanr(reduced_feature_df_scaled_array[:, i], mds_df['Dimension 2'])[0] for i in
                   range(reduced_feature_df_scaled_array.shape[1])],
}, index=reduced_feature_df_scaled.columns)
correlation_matrix = correlation_matrix.round(3)

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt=".3f", annot_kws={"size": 18})
plt.xlabel('Reduced Dimensions', fontsize=18)
plt.ylabel('Original Features', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()

high_impact_features = ['DiscussionNegConsequencesMH_Maybe', 'DiscussionNegConsequencesMH_No',
                        'DiscussionMHDirectSupervisor_Yes', 'HistoryMI_Yes', 'PastMI_Yes', 'CurrentMHDisorder_Yes',
                        'ProfessionalDiagnosis_No', 'ProfessionalDiagnosis_Yes']
for feature in high_impact_features:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Cluster', y=feature, data=df_combined, palette='viridis')
    plt.xlabel('Cluster')
    plt.ylabel(feature)
    plt.show()

grouped_summary = df_combined.groupby('Cluster')['DiscussionNegConsequencesMH_Maybe'].describe()
grouped_summary1 = df_combined.groupby('Cluster')['DiscussionNegConsequencesMH_No'].describe()
grouped_summary2 = df_combined.groupby('Cluster')['DiscussionMHDirectSupervisor_Yes'].describe()
grouped_summary3 = df_combined.groupby('Cluster')['HistoryMI_Yes'].describe()
grouped_summary4 = df_combined.groupby('Cluster')['PastMI_Yes'].describe()
grouped_summary5 = df_combined.groupby('Cluster')['CurrentMHDisorder_Yes'].describe()
grouped_summary6 = df_combined.groupby('Cluster')['ProfessionalDiagnosis_No'].describe()
grouped_summary7 = df_combined.groupby('Cluster')['ProfessionalDiagnosis_Yes'].describe()
print(grouped_summary)
print(grouped_summary1)
print(grouped_summary2)
print(grouped_summary3)
print(grouped_summary4)
print(grouped_summary5)
print(grouped_summary6)
print(grouped_summary7)


#MDS and cluster analysis for unscaled data
df_scaled2 = pd.DataFrame(MinMaxScaler().fit_transform(reduced_feature_df_unscaled),
                          columns=reduced_feature_df_unscaled.columns)
mds = MDS(n_components=2, random_state=42)
mds_transformed2 = mds.fit_transform(df_scaled2)
kmeans2 = KMeans(n_clusters=5, random_state=42)
clusters2 = kmeans2.fit_predict(mds_transformed2)
mds_df2 = pd.DataFrame(mds_transformed2, columns=['Dimension 1', 'Dimension 2'])
df_combined2 = pd.concat([reduced_feature_df_unscaled, mds_df2], axis=1)
df_combined2['Cluster'] = clusters2

inertia = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(mds_transformed2)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

print(df_combined2)
df_combined2.to_csv('C:/Users/sophi/Desktop/ClusterAnalysisUnscaledData.csv', index=False)

plt.scatter(df_combined2['Dimension 1'], df_combined2['Dimension 2'], c=df_combined2['Cluster'], cmap='viridis', s=100)
plt.xlabel('Dimension 1', fontsize=18)
plt.ylabel('Dimension 2', fontsize=18)
plt.colorbar(label='Cluster')
plt.show()

#Projection matrix (Heatmap) for unscaled data
reduced_feature_df_unscaled_array = np.array(df_scaled2)
correlation_matrix2 = pd.DataFrame({
    'Dimension1': [spearmanr(reduced_feature_df_unscaled_array[:, i], mds_df2['Dimension 1'])[0] for i in
                   range(reduced_feature_df_unscaled_array.shape[1])],
    'Dimension2': [spearmanr(reduced_feature_df_unscaled_array[:, i], mds_df2['Dimension 2'])[0] for i in
                   range(reduced_feature_df_unscaled_array.shape[1])],
}, index=reduced_feature_df_unscaled.columns)
correlation_matrix2 = correlation_matrix2.round(3)

sns.heatmap(correlation_matrix2, annot=True, cmap='coolwarm', center=0, fmt=".3f", annot_kws={"size": 18})
plt.xlabel('Reduced Dimensions', fontsize=18)
plt.ylabel('Original Features', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()

#MDS and cluster analysis for feasible data
df_feasible_scaled = pd.DataFrame(MinMaxScaler().fit_transform(df_feasible),
                                  columns=df_feasible.columns)
mds = MDS(n_components=2, random_state=42)
mds_transformed3 = mds.fit_transform(df_feasible_scaled)
kmeans3 = KMeans(n_clusters=4, random_state=42)
clusters3 = kmeans3.fit_predict(mds_transformed3)
mds_df3 = pd.DataFrame(mds_transformed3, columns=['Dimension 1', 'Dimension 2'])
df_combined3 = pd.concat([df_feasible, mds_df3], axis=1)
df_combined3['Cluster'] = clusters3

inertia = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(mds_transformed3)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

print(df_combined3)
df_combined3.to_csv('C:/Users/sophi/Desktop/ClusterAnalysisFeasibleData.csv', index=False)

plt.scatter(df_combined3['Dimension 1'], df_combined3['Dimension 2'], c=df_combined3['Cluster'], cmap='viridis', s=100)
plt.xlabel('Dimension 1', fontsize=18)
plt.ylabel('Dimension 2', fontsize=18)
plt.colorbar(label='Cluster')
plt.show()

#Projection matrix (Heatmap) for feasible data
df_feasible_array = np.array(df_feasible_scaled)
correlation_matrix3 = pd.DataFrame({
    'Dimension1': [spearmanr(df_feasible_array[:, i], mds_df3['Dimension 1'])[0] for i in
                   range(df_feasible_array.shape[1])],
    'Dimension2': [spearmanr(df_feasible_array[:, i], mds_df3['Dimension 2'])[0] for i in
                   range(df_feasible_array.shape[1])],
}, index=df_feasible_scaled.columns)
correlation_matrix2 = correlation_matrix3.round(3)

sns.heatmap(correlation_matrix3, annot=True, cmap='coolwarm', center=0, fmt=".3f", annot_kws={"size": 18})
plt.xlabel('Reduced Dimensions', fontsize=18)
plt.ylabel('Original Features', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()

#analysis of distribution in clusters for feasible data
high_impact_features2 = ['PastMI_No', 'PastMI_Yes', 'CurrentMHDisorder_No', 'CurrentMHDisorder_Yes', 'Gender_Female',
                        'Gender_Male']
for feature in high_impact_features2:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Cluster', y=feature, data=df_combined3, palette='viridis')
    plt.xlabel('Cluster')
    plt.ylabel(feature)
    plt.show()

grouped_summary8 = df_combined3.groupby('Cluster')['PastMI_No'].describe()
grouped_summary9 = df_combined3.groupby('Cluster')['PastMI_Yes'].describe()
grouped_summary10 = df_combined3.groupby('Cluster')['CurrentMHDisorder_No'].describe()
grouped_summary11 = df_combined3.groupby('Cluster')['CurrentMHDisorder_Yes'].describe()
grouped_summary12 = df_combined3.groupby('Cluster')['Gender_Female'].describe()
grouped_summary13 = df_combined3.groupby('Cluster')['Gender_Male'].describe()
print(grouped_summary8)
print(grouped_summary9)
print(grouped_summary10)
print(grouped_summary11)
print(grouped_summary12)
print(grouped_summary13)

#MDS with inverse projection matrix for scaled data
mds_inv = MDS(n_components=2, random_state=42)
mds_transformed_inv = mds_inv.fit_transform(reduced_feature_df_scaled)
projection_matrix = np.linalg.lstsq(mds_transformed_inv, reduced_feature_df_scaled, rcond=None)[0]
inverse_projection_matrix = np.linalg.pinv(projection_matrix)
reconstructed_features = np.dot(mds_transformed_inv, inverse_projection_matrix.T)
reconstructed_df = pd.DataFrame(reconstructed_features, columns=reduced_feature_df_scaled.columns)
reconstruction_error = np.mean((reduced_feature_df_scaled - reconstructed_features) ** 2, axis=0)
print("Reconstruction Error for Each Feature:", reconstruction_error)
print(reconstructed_df)

