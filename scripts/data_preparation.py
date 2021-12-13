import pandas as pd
import numpy as np

from specifications import DemoSpecs, NeuralDataSpecs, OutcomeSpecs, CognitiveSpecs

raw_data = '../raw/'
derived_data = '../derived/'

def extract_cols(filename, period, numeric, rename):
    df = pd.read_csv(f'{raw_data}{filename}', delimiter='\t', low_memory=False)
    if filename in ['srpf01.txt', 'dhx01.txt']:
        df.rename(columns={'visit': 'eventname'}, inplace=True)
    cols = list(rename.keys())
    df = df[['subjectkey', 'eventname'] + cols].iloc[1:, :]
    df = df.rename(columns={'subjectkey': 'subject'})
    df = df.query(f'eventname.str.contains("{period}", na=False)', engine='python').drop(columns=['eventname'])
    for col in cols:
        if col not in numeric:
            df[col] = df[col].astype(str)
            df[col] = [x.replace(',', '.') for x in df[col]]
            df[col] = df[col].astype(float)
            df = df.replace({555: np.nan, 777: np.nan, 888: np.nan, 999: np.nan})
    df = df.rename(columns=rename)
    return df

def get_cognitive_tasks(subjects, period, version, load_latest=False):
    if load_latest:
        cognitive = pd.read_csv(f'{derived_data}{version}/cognitive_{period}.csv')
    else:
        cognitive = pd.DataFrame(subjects, columns=['subject'])
        for spec in CognitiveSpecs:
            df = extract_cols(spec.filename, period, spec.nonnumericcols, spec.cols)
            if spec.filename == 'abcd_sst02.txt':
                df = df.query('sst_acceptable_performance == 1').reset_index(drop=True)
            elif spec.filename == 'abcd_mid01.txt':
                df = df.query('mid_acceptable_performance == 1').reset_index(drop=True)
                df = df.query('mid_num_trials >= 100').reset_index(drop=True)
            elif spec.filename == 'lmtp201.txt':
                df['lmt_efficiency'] = df['lmt_accuracy'] / df['lmt_correct_mrt']
            elif spec.filename == "abcd_mrinback02.txt":
                df = df.query('nb_acceptable_performance == 1').reset_index(drop=True)
            elif spec.filename == "abcd_yddss01.txt":
                df = df.query('ddt_choice_validity > 2').reset_index(drop=True)
            elif spec.filename == 'abcd_yest01.txt':
                df['str_stroop_mrt'] = df['str_mrt_ic'] - df['str_mrt_c']
                df['str_stroop_acc'] = df['str_accuracy_c'] - df['str_accuracy_ic']
            cognitive = pd.merge(cognitive, df, on=['subject'], how='left')

        cognitive.to_csv(f'{derived_data}{version}/cognitive_{period}.csv', index_label=False)
    return cognitive

def get_demographics(subjects, period, version, load_latest=False):
    if load_latest:
        demo = pd.read_csv(f'{derived_data}{version}/demographic_{period}.csv')
    else:
        demo = pd.DataFrame(subjects, columns=['subject'])
        for spec in DemoSpecs:
            df = extract_cols(spec.filename, period, spec.nonnumericcols, spec.cols)
            if spec.filename == 'abcd_ant01.txt':
                df['weight'] = df[['w1', 'w2', 'w3']].mean(axis=1, skipna=True)
                df.drop(columns=['w1', 'w2', 'w3'], inplace=True)

            elif spec.filename == 'abcd_lpmh01.txt':
                df['medhx_p'] = df[["medhx_doctorvisit_p", "medhx_asthma_p", "medhx_allergies_p", "medhx_brain_p",
                                    "medhx_diabetes_p", "medhx_epilepsy_p", "medhx_heart_p", "medhx_headaches_p",
                                    "medhx_emergencyroom_p", "medhx_brokenbones_p", "medhx_seriousinjury_p"]].mean(axis=1)

            elif spec.filename == 'abcd_ssphy01.txt':
                df['puberty_k'] = np.where(df.sex == 'M', df.male_puberty, df.female_puberty)
                df = df[['subject', 'puberty_k']]

            elif spec.filename == 'abcd_fbwpas01.txt':
                df = df.query('fitbit_include == 1').reset_index(drop=True)
                df = df.groupby(['subject']).mean().reset_index(drop=False)


            elif spec.filename == 'neglectful_behavior01.txt':
                df['parent_cares_ss_k'] = df[['parent_care_misbehave_k', 'parent_care_whereabouts_k',
                                              'parent_care_friends_k', 'parent_helphomework_k','parent_safeplay_k',
                                              'parent_gotoschool_k', 'parent_troubleschool_k',
                                              'parent_helpunderstanding_k']].mean(axis=1)

            demo = pd.merge(demo, df, on=['subject'], how='left')
        demo.to_csv(f'{derived_data}{version}/demographic_{period}.csv', index_label=False)
    return demo

def get_neuraldata(subjects, period, version, load_latest=False):
    if load_latest:
        neuraldata = pd.read_csv(f'{derived_data}{version}/neuraldata_{period}.csv')
    else:
        neuraldata = pd.DataFrame(subjects, columns=['subject'])
        for spec in NeuralDataSpecs:
            df = extract_cols(spec.filename, period, spec.nonnumericcols, spec.cols)
            neuraldata = pd.merge(neuraldata, df, on=['subject'], how='left')

        neuraldata.to_csv(f'{derived_data}{version}/neuraldata_{period}.csv', index_label=False)
    return neuraldata

def get_outcomes(subjects, period, version, load_latest=False):
    if load_latest:
        outcomes = pd.read_csv(f'{derived_data}{version}/outcomes_{period}.csv')
    else:
        outcomes = pd.DataFrame(subjects, columns=['subject'])
        for spec in OutcomeSpecs:
            df = extract_cols(spec.filename, period, spec.nonnumericcols, spec.cols)

            if spec.filename == 'emotion_reg_erq_feelings01.txt':
                df['emoreg_sup_ss_k'] = df[['emoreg_sup_control_k', 'emoreg_sup_hide_k', 'emoreg_sup_self_k']].mean(axis=1)
                df['emoreg_reapp_ss_k'] = df[['emoreg_reapp_happy_k', 'emoreg_reapp_less_bad_k',
                                              'emoreg_reapp_thoughts_k']].mean(axis=1)

            outcomes = pd.merge(outcomes, df, on=['subject'], how='left')

        outcomes.to_csv(f'{derived_data}{version}/outcomes_{period}.csv', index_label=False)

    return outcomes
