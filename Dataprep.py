import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# read csv
df = pd.read_csv('C:/Users/sophi/PycharmProjects/MentalHealth/.venv/mental-heath-in-tech-2016_20161114.csv')
print(df.columns)

# delete unncessary observations, but answer categories together, and rename variables
df = df[df['Are you self-employed?'] == 0]
df = df[(df['Is your employer primarily a tech company/organization?'] == 1) | (df['Is your primary role within your company related to tech/IT?'] == 1)]
df = df[(df['Does your employer provide mental health benefits as part of healthcare coverage?'] != 'No')]
df = df[df['Do you know the options for mental health care available under your employer-provided coverage?'].isin(['I am not sure', 'N/A', 'No', 'Yes'])]
df = df[df['Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?'].isin(['I don\'t know', 'N/A', 'No', 'Yes'])]
df = df[df['Does your employer offer resources to learn more about mental health concerns and options for seeking help?'].isin(['I don\'t know', 'No', 'Yes'])]
df = df[df['Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?'].isin(['I don\'t know', 'No', 'Yes'])]
df = df[df['Do you think that discussing a mental health disorder with your employer would have negative consequences?'].isin(['Maybe', 'Yes', 'No'])]
df = df[df['Do you think that discussing a physical health issue with your employer would have negative consequences?'].isin(['Maybe', 'Yes', 'No'])]
df = df[df['Would you feel comfortable discussing a mental health disorder with your coworkers?'].isin(['Maybe', 'Yes', 'No'])]
df = df[df['Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?'].isin(['Maybe', 'Yes', 'No'])]
df = df[df['Do you feel that your employer takes mental health as seriously as physical health?'].isin(['I don\'t know', 'No', 'Yes'])]
df = df[df['Have you heard of or observed negative consequences for co-workers who have been open about mental health issues in your workplace?'].isin(['No', 'Yes'])]
df = df[~df['What is your age?'].isin([3, 323])]
df['UnsupportiveResponseMH'] = df['Have you observed or experienced an unsupportive or badly handled response to a mental health issue in your current or previous workplace?'].replace({'N/A': 'Maybe/Not sure'})
df['NegatveImpactRevealMH'] = df['Have your observations of how another individual who discussed a mental health disorder made you less likely to reveal a mental health issue yourself in your current workplace?'].replace({'N/A': 'Maybe'})
df['Gender'] = df['What is your gender?'].replace({'Female': 'Female', 'AFAB': 'Female', 'Agender': 'Diverse', 'Bigender': 'Diverse', 'Androgynous': 'Diverse', 'Cis Male': 'Male', 'Cis female': 'Female',
                                                   'Cis male': 'Male', 'Cis-woman': 'Female', 'Cisgender Female': 'Female', 'Dude': 'Male', 'Enby': 'Diverse', 'F': 'Female', 'Female ': 'Female', 'Female assigned at birth ': 'Female',
                                                   'Female (props for making this a freeform field, though)': 'Female', 'Female or Multi-Gender Femme': 'Female', 'Fluid': 'Diverse', 'Genderfluid': 'Diverse',
                                                   'Genderfluid (born female)': 'Diverse', 'Genderqueer': 'Diverse', 'I identify as female.': 'Female', 'I\'m a man why didn\'t you make this a drop down question. You should of asked sex? And I would of answered yes please. Seriously how much text can this take? ': 'Male',
                                                   'M': 'Male', 'MALE': 'Male', 'Male ': 'Male', 'Male (cis)': 'Male', 'Male (trans, FtM)': 'Male',
                                                   'Male.': 'Male', 'Male/genderqueer': 'Diverse', 'Malr': 'Male', 'Man': 'Male', 'M|': 'Male', 'Nonbinary': 'Diverse', 'Other': 'Diverse', 'Other/Transfeminine': 'Diverse',
                                                   'Queer': 'Diverse', 'Sex is male': 'Male', 'Transgender woman': 'Diverse', 'Transitioned, M2F': 'Diverse', 'Unicorn': 'Diverse', 'Woman': 'Female', 'cis male': 'Male', 'cis man': 'Male',
                                                   'cisdude': 'Male', 'f': 'Female', 'fem': 'Female', 'female': 'Female', 'female ': 'Female', 'female-bodied; no feelings about gender': 'Female', 'female/woman': 'Female',
                                                   'fm': 'Female', 'genderqueer': 'Diverse', 'genderqueer woman': 'Diverse', 'human': 'Diverse', 'm': 'Male', 'mail': 'Male', 'male': 'Male', 'male ': 'Male', 'male 9:1 female, roughly': 'Male', 'man': 'Male', 'mtf': 'Diverse', 'nb masculine': 'Diverse',
                                                   'non-binary': 'Diverse', 'none of your business': 'Diverse', 'woman': 'Female', 'Human': 'Diverse', 'Genderflux demi-girl': 'Diverse'})
df = df.rename(columns={'Do you know the options for mental health care available under your employer-provided coverage?': 'KnowledgeMHCEmployerCoverage',
                        'Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?': 'DiscussionMHEmployer',
                        'Does your employer offer resources to learn more about mental health concerns and options for seeking help?': 'EmployerResourcesMH',
                        'Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?': 'AnonymityProtection',
                        'Do you think that discussing a mental health disorder with your employer would have negative consequences?': 'DiscussionNegConsequencesMH',
                        'Do you think that discussing a physical health issue with your employer would have negative consequences?': 'DiscussionNegConsequencesPH',
                        'Would you feel comfortable discussing a mental health disorder with your coworkers?': 'DiscussionMHCoworkers',
                        'Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?': 'DiscussionMHDirectSupervisor',
                        'Do you feel that your employer takes mental health as seriously as physical health?': 'EmployerMHSerious',
                        'Have you heard of or observed negative consequences for co-workers who have been open about mental health issues in your workplace?': 'NegConsequencesOpenMH',
                        'Do you have previous employers?': 'PreviousEmployers',
                        'Were you aware of the options for mental health care provided by your previous employers?': 'AwarenessMHCarePrevEmployers',
                        'Would you be willing to bring up a physical health issue with a potential employer in an interview?': 'PHIssueInterview',
                        'Would you bring up a mental health issue with a potential employer in an interview?': 'MHIssuesInterview',
                        'Do you feel that being identified as a person with a mental health issue would hurt your career?': 'MHHurtCareer',
                        'Do you think that team members/co-workers would view you more negatively if they knew you suffered from a mental health issue?': 'NegViewedByTeam',
                        'Do you have a family history of mental illness?': 'HistoryMI',
                        'Have you had a mental health disorder in the past?': 'PastMI',
                        'Do you currently have a mental health disorder?': 'CurrentMHDisorder',
                        'Have you been diagnosed with a mental health condition by a medical professional?': 'ProfessionalDiagnosis',
                        'Have you ever sought treatment for a mental health issue from a mental health professional?': 'ProfTreatmentMHIssue',
                        'What is your age?': 'Age'})

def map_continent(country):
    Asia = ['Afghanistan', 'Bangladesh', 'China', 'India', 'Israel', 'Japan', 'Pakistan', 'Taiwan', 'Vietnam']
    Africa = ['Algeria', 'South Africa']
    South_America = ['Argentina', 'Brazil', 'Chile', 'Colombia', 'Costa Rica', 'Ecuador', 'Guatemala',  'Venezuela']
    Oceania = ['Australia', 'New Zealand']
    Europe = ['Austria', 'Belgium', 'Bosnia and Herzegovina', 'Bulgaria', 'Czech Republic', 'Denmark', 'Estonia', 'Finland',
              'France', 'Germany', 'Greece', 'Hungary', 'Ireland', 'Italy', 'Lithuania', 'Netherlands', 'Norway', 'Poland',
              'Romania', 'Russia', 'Serbia', 'Slovakia', 'Spain', 'Sweden', 'Switzerland', 'United Kingdom']
    North_America = ['Mexico','Canada', 'United States of America']

    if country in Asia:
        return 'Asia'
    elif country in Africa:
        return 'Africa'
    elif country in South_America:
        return 'South America'
    elif country in Oceania:
        return 'Oceania'
    elif country in Europe:
        return 'Europe'
    elif country in North_America:
        return 'North America'
    else:
        return 'Other'

df['ContinentLive'] = df['What country do you live in?'].apply(map_continent)
df['ContinentWork'] = df['What country do you work in?'].apply(map_continent)

#encode string variables
#ordinal variables
category_mapping1 = {'Never': 1, 'Sometimes': 2, 'Always': 3}
df['RemoteWork'] = df['Do you work remotely?'].map(category_mapping1)
category_mapping2 = {'1-5': 1, '6-25': 2, '26-100': 3, '100-500': 4, '500-1000': 5, 'More than 1000': 6}
df['NumberOfEmployees'] = df['How many employees does your company or organization have?'].map(category_mapping2)
category_mapping3 = {'I don\'t know': 0, 'Neither easy nor difficult': 1, 'Somewhat difficult': 2, 'Somewhat easy': 3, 'Very difficult': 4, 'Very easy': 5}
df['RequestMedicalLeave'] = df['If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:'].map(category_mapping3)
category_mapping4 = {'I don\'t know': 0, 'No, none did': 1, 'Some did': 2, 'Yes, they all did': 3}
df['MHBenefitsPrevEmployers'] = df['Have your previous employers provided mental health benefits?'].map(category_mapping4)
category_mapping5 = {'I don\'t know': 0, 'None did': 1, 'Some did': 2, 'Yes, they all did': 3}
df['DiscussionMHPrevEmployers'] = df['Did your previous employers ever formally discuss mental health (as part of a wellness campaign or other official communication)?'].map(category_mapping5)
category_mapping6 = {'None did': 1, 'Some did': 2, 'Yes, they all did': 3}
df['ResourcesMHPrevEmployers'] = df['Did your previous employers provide resources to learn more about mental health issues and how to seek help?'].map(category_mapping6)
category_mapping7 = {'I don\'t know': 0, 'No': 1, 'Sometimes': 2, 'Yes, always': 3}
df['ProtectionAnonymityPrevEmployers'] = df['Was your anonymity protected if you chose to take advantage of mental health or substance abuse treatment resources with previous employers?'].map(category_mapping7)
category_mapping8 = {'I don\'t know': 0, 'None of them': 1, 'Some of them': 2, 'Yes, all of them': 3}
df['NegConsequencesDiscussionMHPrevEmployer'] = df['Do you think that discussing a mental health disorder with previous employers would have negative consequences?'].map(category_mapping8)
category_mapping9 = {'None of them': 1, 'Some of them': 2, 'Yes, all of them': 3}
df['NegConsequencesDiscussionPHPrevEmployer'] = df['Do you think that discussing a physical health issue with previous employers would have negative consequences?'].map(category_mapping9)
category_mapping10 = {'No, at none of my previous employers': 1, 'Some of my previous employers': 2, 'Yes, at all of my previous employers': 3}
df['DiscussionMHPrevCoworker'] = df['Would you have been willing to discuss a mental health issue with your previous co-workers?'].map(category_mapping10)
category_mapping11 = {'I don\'t know': 0, 'No, at none of my previous employers': 1, 'Some of my previous employers': 2, 'Yes, at all of my previous employers': 3}
df['DiscussionMHPrevSupervisor'] = df['Would you have been willing to discuss a mental health issue with your direct supervisor(s)?'].map(category_mapping11)
df['SeriousnessMHPrevEmployer'] = df['Did you feel that your previous employers took mental health as seriously as physical health?'].map(category_mapping5)
df['NegConsequencesCoworkersPrevEmployer'] = df['Did you hear of or observe negative consequences for co-workers with mental health issues in your previous workplaces?'].map(category_mapping9)
category_mapping12 = {'Not applicable to me (I do not have a mental illness)': 0, 'Not open at all': 1, 'Somewhat not open': 2, 'Neutral': 3, 'Somewhat open': 4, 'Very open': 5}
df['ShareMI'] = df['How willing would you be to share with friends and family that you have a mental illness?'].map(category_mapping12)
category_mapping13 = {'Not applicable to me': 0, 'Never': 1, 'Rarely': 2, 'Sometimes': 3, 'Often': 4}
df['MHInterferenceWorkEffTreat'] = df['If you have a mental health issue, do you feel that it interferes with your work when being treated effectively?'].map(category_mapping13)
df['MHInterferenceWorkNoTreat'] = df['If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?'].map(category_mapping13)

#nominal variables
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
ohetransform1 = ohe.fit_transform(df[['KnowledgeMHCEmployerCoverage']])
df = pd.concat([df, ohetransform1], axis = 1).drop(columns = ['KnowledgeMHCEmployerCoverage'])
ohetransform2 = ohe.fit_transform(df[['DiscussionMHEmployer']])
df = pd.concat([df, ohetransform2], axis = 1).drop(columns = ['DiscussionMHEmployer'])
ohetransform3 = ohe.fit_transform(df[['EmployerResourcesMH']])
df = pd.concat([df, ohetransform3], axis = 1).drop(columns = ['EmployerResourcesMH'])
ohetransform4 = ohe.fit_transform(df[['AnonymityProtection']])
df = pd.concat([df, ohetransform4], axis = 1).drop(columns = ['AnonymityProtection'])
ohetransform5 = ohe.fit_transform(df[['DiscussionNegConsequencesMH']])
df = pd.concat([df, ohetransform5], axis = 1).drop(columns = ['DiscussionNegConsequencesMH'])
ohetransform6 = ohe.fit_transform(df[['DiscussionNegConsequencesPH']])
df = pd.concat([df, ohetransform6], axis = 1).drop(columns = ['DiscussionNegConsequencesPH'])
ohetransform7 = ohe.fit_transform(df[['DiscussionMHCoworkers']])
df = pd.concat([df, ohetransform7], axis = 1).drop(columns = ['DiscussionMHCoworkers'])
ohetransform8 = ohe.fit_transform(df[['DiscussionMHDirectSupervisor']])
df = pd.concat([df, ohetransform8], axis = 1).drop(columns = ['DiscussionMHDirectSupervisor'])
ohetransform9 = ohe.fit_transform(df[['EmployerMHSerious']])
df = pd.concat([df, ohetransform9], axis = 1).drop(columns = ['EmployerMHSerious'])
ohetransform10 = ohe.fit_transform(df[['NegConsequencesOpenMH']])
df = pd.concat([df, ohetransform10], axis = 1).drop(columns = ['NegConsequencesOpenMH'])
ohetransform11 = ohe.fit_transform(df[['AwarenessMHCarePrevEmployers']])
df = pd.concat([df, ohetransform11], axis = 1).drop(columns = ['AwarenessMHCarePrevEmployers'])
ohetransform12 = ohe.fit_transform(df[['PHIssueInterview']])
df = pd.concat([df, ohetransform12], axis = 1).drop(columns = ['PHIssueInterview'])
ohetransform13 = ohe.fit_transform(df[['MHIssuesInterview']])
df = pd.concat([df, ohetransform13], axis = 1).drop(columns = ['MHIssuesInterview'])
ohetransform14 = ohe.fit_transform(df[['MHHurtCareer']])
df = pd.concat([df, ohetransform14], axis = 1).drop(columns = ['MHHurtCareer'])
ohetransform15 = ohe.fit_transform(df[['NegViewedByTeam']])
df = pd.concat([df, ohetransform15], axis = 1).drop(columns = ['NegViewedByTeam'])
ohetransform16 = ohe.fit_transform(df[['UnsupportiveResponseMH']])
df = pd.concat([df, ohetransform16], axis = 1).drop(columns = ['UnsupportiveResponseMH'])
ohetransform17 = ohe.fit_transform(df[['NegatveImpactRevealMH']])
df = pd.concat([df, ohetransform17], axis = 1).drop(columns = ['NegatveImpactRevealMH'])
ohetransform18 = ohe.fit_transform(df[['HistoryMI']])
df = pd.concat([df, ohetransform18], axis = 1).drop(columns = ['HistoryMI'])
ohetransform19 = ohe.fit_transform(df[['PastMI']])
df = pd.concat([df, ohetransform19], axis = 1).drop(columns = ['PastMI'])
ohetransform20 = ohe.fit_transform(df[['CurrentMHDisorder']])
df = pd.concat([df, ohetransform20], axis = 1).drop(columns = ['CurrentMHDisorder'])
ohetransform21 = ohe.fit_transform(df[['ProfessionalDiagnosis']])
df = pd.concat([df, ohetransform21], axis = 1).drop(columns = ['ProfessionalDiagnosis'])
ohetransform22 = ohe.fit_transform(df[['Gender']])
df = pd.concat([df, ohetransform22], axis = 1).drop(columns = ['Gender'])
ohetransform23 = ohe.fit_transform(df[['ContinentLive']])
df = pd.concat([df, ohetransform23], axis = 1).drop(columns = ['ContinentLive'])
ohetransform24 = ohe.fit_transform(df[['ContinentWork']])
df = pd.concat([df, ohetransform24], axis = 1).drop(columns = ['ContinentWork'])

#string variables
df['LeadershipPosition'] = df['Which of the following best describes your work position?'].str.contains('Lead', case=False, na=False).astype(int)

df['SupportPosition'] = df['Which of the following best describes your work position?'].str.contains('Support', case=False, na=False).astype(int)

MIcategories = {
    "Anxiety Disorders": [
        "Anxiety Disorder (Generalized, Social, Phobia, etc)",
        "Anxiety Disorder (Generalized, Social, Phobia, etc)|Substance Use Disorder",
        "Anxiety Disorder (Generalized, Social, Phobia, etc)|Obsessive-Compulsive Disorder",
        "Anxiety Disorder (Generalized, Social, Phobia, etc)|Mood Disorder (Depression, Bipolar Disorder, etc)",
        "Anxiety Disorder (Generalized, Social, Phobia, etc)|Attention Deficit Hyperactivity Disorder",
        "Anxiety Disorder (Generalized, Social, Phobia, etc)|Post-traumatic Stress Disorder",
        "Anxiety Disorder (Generalized, Social, Phobia, etc)|Personality Disorder (Borderline, Antisocial, Paranoid, etc)",
        "Anxiety Disorder (Generalized, Social, Phobia, etc)|Mood Disorder (Depression, Bipolar Disorder, etc)|Substance Use Disorder",
        "Anxiety Disorder (Generalized, Social, Phobia, etc)|Eating Disorder (Anorexia, Bulimia, etc)|Posttraumatic stress disorder",
    ],
    "Mood Disorders": [
        "Mood Disorder (Depression, Bipolar Disorder, etc)",
        "Seasonal Affective Disorder",
        "Mood Disorder (Depression, Bipolar Disorder, etc)|Attention Deficit Hyperactivity Disorder",
        "Mood Disorder (Depression, Bipolar Disorder, etc)|Suicidal Ideation",
        "Mood Disorder (Depression, Bipolar Disorder, etc)|Post-traumatic Stress Disorder",
        "Mood Disorder (Depression, Bipolar Disorder, etc)|Stress Response Syndromes",
        "Mood Disorder (Depression, Bipolar Disorder, etc)|Personality Disorder (Borderline, Antisocial, Paranoid, etc)",
    ],
    "Eating Disorders": [
        "Eating Disorder (Anorexia, Bulimia, etc)",
        "Mood Disorder (Depression, Bipolar Disorder, etc)|Eating Disorder (Anorexia, Bulimia, etc)",
        "Anxiety Disorder (Generalized, Social, Phobia, etc)|Mood Disorder (Depression, Bipolar Disorder, etc)|Eating Disorder (Anorexia, Bulimia, etc)",
    ],
    "Attention Deficit Disorders": [
        "Attention Deficit Hyperactivity Disorder",
        "ADD (w/o Hyperactivity)",
        "Mood Disorder (Depression, Bipolar Disorder, etc)|Attention Deficit Hyperactivity Disorder",
    ],
    "Autism Spectrum Disorders": [
        "Autism (Asperger's)",
        "Autism Spectrum Disorder",
        "Autism - while not a 'mental illness', still greatly affects how I handle anxiety",
        "Asperger Syndrome",
    ],
    "Obsessive-Compulsive and Related Disorders": [
        "Obsessive-Compulsive Disorder",
        "Obsessive-Compulsive Disorder|Substance Use Disorder",
        "Anxiety Disorder (Generalized, Social, Phobia, etc)|Obsessive-Compulsive Disorder",
    ],
    "Personality Disorders": [
        "Personality Disorder (Borderline, Antisocial, Paranoid, etc)",
        "Schizotypal Personality Disorder",
    ],
    "Post-Traumatic and Stress-Related Disorders": [
        "Post-traumatic Stress Disorder",
        "Stress Response Syndromes",
        "Anxiety Disorder (Generalized, Social, Phobia, etc)|Post-traumatic Stress Disorder",
    ],
    "Psychotic Disorders": [
        "Psychotic Disorder (Schizophrenia, Schizoaffective, etc)"
    ],
    "Substance Use and Addictive Disorders": [
        "Substance Use Disorder",
        "Addictive Disorder",
        "Substance Use Disorder|Addictive Disorder",
    ],
}

def assign_category(answer):
    if isinstance(answer, str):
        for category, keywords in MIcategories.items():
            if any(keyword in answer for keyword in keywords):
                return category
    return ""

# Neue Spalte mit den Kategorien
df["MI"] = df["If so, what condition(s) were you diagnosed with?"].fillna("")
df["MI"] = df["If so, what condition(s) were you diagnosed with?"].apply(assign_category)
ohetransform25 = ohe.fit_transform(df[['MI']])
df = pd.concat([df, ohetransform25], axis = 1).drop(columns = ['MI'])

column_list = df.columns.tolist()
print(column_list)

df.to_csv('C:/Users/sophi/Desktop/MentalHealthClean.csv', index=False)
