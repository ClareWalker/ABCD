# Checks for acceptable performance on cognitive tasks using trial by trial data

import pandas as pd
import numpy as np
import glob

raw_data = '../raw/'
derived_data = '../derived/'

trials_sst_go=300
trials_sst_stop=60
trials_lmt=32
trials_est=96
trials_mid=100
trials_nb=160

est_cutoffs = (0.75, 200, 0.3, 0.95)         # lb_complete, lb_rt, lb_acc, ub_favkey
sst_cutoffs = (0.75, 0.5, 200, 0.3, 0.1, 0.95)    # lb_complete, lb_complete_stop, lb_rt, lb_acc, lb_acc_stop, ub_favkey
mid_cutoffs = (0.75, 200, 0.3)               # lb_complete, lb_rt, lb_acc
lmt_cutoffs = (0.75, 200, 0.3, 0.95)         # lb_complete, lb_rt, lb_acc, ub_favkey
nb_cutoffs = (0.75, 200, 0.3, 0.95)          # lb_complete, lb_rt, lb_acc, ub_favkey


def check_task(folder, period, shortname, check_func):
    cols = ['subject', shortname]
    df = pd.DataFrame(columns=cols)
    files = glob.glob(f'{raw_data}{folder}/*{period}*zip')
    for file in files:
        row = pd.Series(check_func(file), index=cols)
        df = df.append(row, ignore_index=True)

    return df

def check_ddt(file):
    valid = True
    df = pd.read_csv(file)
    subject = df.subject[0]
    df = df.query('ddis_trialtype == "test" and ddis_choicelatency_ms > 0').reset_index(drop=True)
    df = df.sort_values(by=['ddis_delay_indays', 'trial']).drop_duplicates(subset=['ddis_delay'], keep='last')

    svs = df.ddis_indifferencepoint.values / 100
    delays = df.ddis_delay_indays.values

    if len(delays) < 5:
        valid = False

    elif np.isfinite(svs).sum() < 5:
        valid = False

    elif delays[0] != 0.25 or delays[-1] != 1826.25:
        valid = False

    elif svs[0] < svs[-1]:
        valid = False

    return subject, valid


def check_nb(file):
    valid = True
    df = pd.read_csv(file)

    lb_complete, lb_rt, lb_accuracy, ub_favkey = nb_cutoffs
    try:
        subject = df.subject[0]
        df = df.query('enback_stim_rt > 0').reset_index(drop=True)
        trials = df.shape[0]
        rt = df.enback_stim_rt.median()
        accuracy = df.enback_stim_acc.mean()
        favkey = df.query(f'enback_stim_resp == "{df.enback_stim_resp.mode()[0]}"').shape[0] / trials

        if trials < lb_complete * trials_mid:
            valid = False
        if rt < lb_rt:
            valid = False
        if accuracy < lb_accuracy:
            valid = False
        if favkey >= ub_favkey:
            valid = False

    except:
        valid = False

    return subject, valid


def check_mid(file):
    valid = True
    df = pd.read_csv(file)
    subject = 'NDAR_' + file.split('/')[-1].split('_')[1]
    lb_complete, lb_rt, lb_accuracy = mid_cutoffs
    try:
        trials = df.shape[0]
        rt = df.mid_rt.median()
        accuracy = df.mid_acc.mean()

        if trials < lb_complete * trials_mid:
            valid = False
        if rt < lb_rt:
            valid = False
        if accuracy < lb_accuracy:
            valid = False

    except:
        valid = False

    return subject, valid


def check_sst(file):
    valid = True
    df = pd.read_csv(file)
    subject = df.subject[0]
    lb_complete, lb_complete_stops, lb_rt, lb_accuracy, lb_accuracy_stops, ub_favkey = sst_cutoffs

    try:
        go = df.query('sst_expcon == "GoTrial" and sst_go_rt > 0').reset_index(drop=True)
        stop = df.query('sst_expcon == "VariableStopTrial"').reset_index(drop=True)
        stop = stop.query('sst_ssd_dur > 200 and sst_ssd_dur <= 700').reset_index(drop=True)

        go_trials = go.shape[0]
        stop_trials = stop.shape[0]
        rt = go.sst_go_rt.median()
        go_accuracy = go.sst_choiceacc.mean()
        stop_accuracy = stop.sst_inhibitacc.mean()
        favkey = go.query(f'sst_primaryresp == "{go.sst_primaryresp.mode()[0]}"').shape[0] / (go_trials + stop_trials)

        failed_stops_mrt = stop.query('sst_inhibitacc == 0').sst_stopsignal_rt.mean()
        go_mrt = go.sst_go_rt.mean()

        if failed_stops_mrt > go_mrt:
            valid = False
        if go_trials < lb_complete * trials_sst_go:
            valid = False
        if stop_trials < lb_complete * trials_sst_stop:
            valid = False
        if rt < lb_rt:
            valid = False
        if go_accuracy < lb_accuracy:
            valid = False
        if stop_accuracy < lb_accuracy_stops:
            valid = False
        if favkey >= ub_favkey:
            valid = False

    except:
        valid = False

    return subject, valid


def check_est(file):
    valid = True
    df = pd.read_csv(file)
    subject = df.subject[0]
    lb_complete, lb_rt, lb_accuracy, ub_favkey = est_cutoffs

    try:
        df = df.query('blockcode == "testMC" or blockcode == "testE"').reset_index(drop=True)
        df = df.query('latency > 0').reset_index(drop=True)
        trials = df.shape[0]
        rt = df.latency.median()
        accuracy = df.correct.mean()
        favkey = df.query(f'response == "{df.response.mode()[0]}"').shape[0] / trials

        if trials < lb_complete * trials_est:
            valid = False
        if rt < lb_rt:
            valid = False
        if accuracy < lb_accuracy:
            valid = False
        if favkey >= ub_favkey:
            valid = False
    except:
        valid = False

    return subject, valid


def check_lmt(file):
    valid = True
    df = pd.read_csv(file)
    subject = df.subject[0]
    lb_complete, lb_rt, lb_accuracy, ub_favkey = lmt_cutoffs
    try:
        if len(df.lmt_trialcode.unique()) == 1:
            df = df.query('lmt_blockcode == "test" and lmt_latency > 0').reset_index(drop=True)
        else:
            df = df.query(
                'lmt_blockcode == "test" and lmt_trialcode == "littleManPresentation" and lmt_latency > 0').reset_index(
                drop=True)

        trials = df.shape[0]
        rt = df.lmt_latency.median()
        accuracy = df.lmt_correct.mean()
        favkey = df.query(f'lmt_response == "{df.lmt_response.mode()[0]}"').shape[0] / trials

        if trials < lb_complete * trials_lmt:
            valid = False
        if rt < lb_rt:
            valid = False
        if accuracy < lb_accuracy:
            valid = False
        if favkey >= ub_favkey:
            valid = False

    except:
        valid = False

    return subject, valid