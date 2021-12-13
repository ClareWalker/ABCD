# DV caluculation from trial by trial data

import pandas as pd
import numpy as np
import glob
from math import floor, ceil

raw_data = '../raw/'
derived_data = '../derived/'

lmt_cols = ['subject', 'lmt_acc', 'lmt_rt', 'lmt_efficiency']
sst_cols = ['subject', 'sst_ssrt']
mid_cols = ['subject', 'mid_hit_rate', 'mid_rt', 'mid_payout', 'mid_loss_aversion_rt', 'mid_incentive_rt',
            'mid_loss_aversion_acc', 'mid_incentive_acc']
est_cols = ['subject', 'est_stroop_control_rt', 'est_stroop_control_acc', 'est_stroop_control_crt',
            'est_proactive_control_crt', 'est_reactive_control_crt']
ddt_cols = ['subject', 'ddt_choice_latency', 'ddt_area']
nb_cols = ['subject', 'nb_performance', 'nb_regulation']


def calc_DV(folder, period, shortname, dv_func, valid):
    df = pd.DataFrame(columns=eval(shortname + '_cols'))

    files = glob.glob(f'{raw_data}{folder}/*{period}*.zip')
    n = len(files)
    i = 1
    for file in files:
        filename = file.split('/')[-1]
        subject = filename.split('_')[0] + '_' + filename.split('_')[1]

        file = pd.read_csv(file)
        if valid.query(f'subject == "{subject}"').shape[0] > 0:
            if valid.query(f'subject == "{subject}"')[shortname].values[0]:
                row = dv_func(file)
                df = df.append(row, ignore_index=True)

        if (i % 1000) == 0:
            print(f'Progress for {shortname}: {i} out of {n}')
        i += 1

    return df

def calc_cct_DV(df, period):
    df = df[['subjectkey', 'eventname', 'cash_choice_task']].iloc[1:, :]
    df = df.rename(columns={'subjectkey': 'subject'})
    df = df.query(f'eventname.str.contains("{period}")', engine='python').drop(columns=['eventname'])
    df['cct'] = np.where(df.cash_choice_task == "1", 0, np.where(df.cash_choice_task == "2", 1, np.nan))
    df.drop(['cash_choice_task'], axis=1, inplace=True)

    return df


def calc_lmt_DV(df):
    subject = df.subject[0]

    if df.shape[1] < 10:
        df = df.query('lmt_latency > 0').reset_index(drop=True)

    elif len(df.lmt_trialcode.unique()) > 1:
        df = df.query(
            'lmt_blockcode == "test" and lmt_trialcode == "littleManPresentation" and lmt_latency > 0').reset_index(
            drop=True)
    else:
        df = df.query('lmt_blockcode == "test" and lmt_latency > 0').reset_index(drop=True)

    acc = df.lmt_correct.mean()
    rt = df.query('lmt_correct == 1').lmt_latency.mean() / 1000
    efficiency = acc / rt

    return pd.Series([subject, acc, rt, efficiency], index=lmt_cols)


def calc_est_DV(df):
    subject = df.subject[0]

    df = df.query('blockcode == "testMC" or blockcode == "testE"').reset_index(drop=True)
    df = df.query('latency > 0').reset_index(drop=True)

    congruent = df[df['values.congruence'] == 1].copy()
    incongruent = df[df['values.congruence'] == 2].copy()

    stroop_acc = incongruent.correct.mean() - congruent.correct.mean()
    stroop_rt = congruent.latency.mean() - incongruent.latency.mean()
    stroop_crt = congruent.query('correct == 1').latency.mean() - incongruent.query('correct == 1').latency.mean()

    proactive_crt = congruent.query('blockcode == "testMC" and correct == 1').latency.mean() - incongruent.query(
        'blockcode == "testMC" and correct == 1').latency.mean()
    reactive_crt = congruent.query('blockcode == "testE" and correct == 1').latency.mean() - incongruent.query(
        'blockcode == "testE" and correct == 1').latency.mean()

    return pd.Series([subject, stroop_rt, stroop_acc, stroop_crt, proactive_crt, reactive_crt], index=est_cols)

def calc_ddt_DV(df):
    subject = df.subject[0]
    df = df.query('ddis_trialtype == "test" and ddis_choicelatency_ms > 0').reset_index(drop=True)

    choice_latency = df.ddis_choicelatency_ms.mean() / 1000

    df = df.sort_values(by=['ddis_delay_indays', 'trial']).drop_duplicates(subset=['ddis_delay'], keep='last')

    svs = df.ddis_indifferencepoint.values / 100
    delays = df.ddis_delay_indays.values
    delays = delays / np.nanmax(delays)

    area = 0
    for i in range(len(delays) - 1):
        area += (delays[i + 1] - delays[i]) * np.nanmean([svs[i], svs[i + 1]])

    return pd.Series([subject, choice_latency, area], index=ddt_cols)


def calc_sst_DV(df):
    subject = df.subject[0]

    go = df.query('sst_expcon == "GoTrial" and sst_go_rt > 0').reset_index(drop=True)
    stop = df.query('sst_expcon == "VariableStopTrial"').reset_index(drop=True)
    stop = stop.query('sst_ssd_dur > 200 and sst_ssd_dur <= 700').reset_index(drop=True)

    # calcuate SSRT and SSD
    go_sorted = go.sst_go_rt.sort_values(ascending=True)
    prob_stop_failure = (1 - stop.sst_inhibitacc.mean())
    index = prob_stop_failure * (len(go_sorted) - 1)
    index = [floor(index), ceil(index)]
    ssd = stop.sst_ssd_dur.mean()
    ssrt = -1 * max(go_sorted.iloc[index].mean() - ssd, 0)

    return pd.Series([subject, ssrt], index=sst_cols)


def calc_mid_DV(df):
    subject = df.subject[0]

    hit_rate = df.mid_acc.mean()
    payout = df.mid_trialmoney.sum()

    correct = df.query('mid_acc == 1').reset_index(drop=True)
    rt = correct.mid_rt.mean()
    neutral_rt = correct.query('mid_anticipationtype == "Neutral"').mid_rt.mean()
    money_rt = correct.query('mid_anticipationtype != "Neutral"').mid_rt.mean()
    reward_rt = correct.query('mid_anticipationtype == "SR" or mid_anticipationtype == "LR"').mid_rt.mean()
    loss_rt = correct.query('mid_anticipationtype == "SL" or mid_anticipationtype == "LL"').mid_rt.mean()

    if reward_rt != 0 and loss_rt != 0:
        loss_aversion_rt = loss_rt / reward_rt  # bias towards avoiding loss vs gaining rewards (>1 means responed faster to loss aversion)
    else:
        loss_aversion_rt = np.nan

    if money_rt != 0 and neutral_rt != 0:
        incentive_rt = money_rt / neutral_rt  # bias to incencentivized trials over neutral ones (>1 means responded faster to incentivised trials)
    else:
        incentive_rt = np.nan

    neutral_acc = df.query('mid_anticipationtype == "Neutral"').mid_acc.mean()
    money_acc = df.query('mid_anticipationtype != "Neutral"').mid_acc.mean()
    reward_acc = df.query('mid_anticipationtype == "SR" or mid_anticipationtype == "LR"').mid_acc.mean()
    loss_acc = df.query('mid_anticipationtype == "SL" or mid_anticipationtype == "LL"').mid_acc.mean()

    if reward_acc != 0 and loss_acc != 0:
        loss_aversion_acc = loss_acc / reward_acc  # bias towards avoiding loss vs gaining rewards (>1 means responed faster to loss aversion)
    else:
        loss_aversion_acc = np.nan

    if money_acc != 0 and neutral_acc != 0:
        incentive_acc = money_acc / neutral_acc  # bias to incencentivized trials over neutral ones (>1 means responded faster to incentivised trials)
    else:
        incentive_acc = np.nan

    return pd.Series([subject, hit_rate, rt, payout, loss_aversion_rt, incentive_rt, loss_aversion_acc, incentive_acc],
                     index=mid_cols)


def calc_nb_DV(df):
    subject = df.subject[0]
    df = df.query('enback_stim_rt > 0')

    hit_rate = df.query('enback_loadcon == "2-Back" and enback_targettype == "target"').enback_stim_acc.mean()
    fa_rate = 1 - df.query('enback_loadcon == "2-Back" and enback_targettype == "lure"').enback_stim_acc.mean()

    performance = hit_rate - fa_rate

    neutral_acc = df.query(
        'enback_stimtype == "NeutFace" and enback_targettype == "target"').enback_stim_acc.mean()  # control
    fearful_acc = df.query('enback_stimtype == "NegFace" and enback_targettype == "target"').enback_stim_acc.mean()
    neutral_rt = df.query(
        'enback_stimtype == "NeutFace" and enback_targettype == "target"').enback_stim_rt.mean() / 1000  # control
    fearful_rt = df.query('enback_stimtype == "NegFace" and enback_targettype == "target"').enback_stim_rt.mean() / 1000

    regulation = (fearful_acc / fearful_rt - neutral_acc / neutral_rt)

    return pd.Series([subject, performance, regulation], index=nb_cols)
