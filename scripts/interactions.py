# Calculation of interaction terms, each specified by a function

import numpy as np

def add_interactions(df, lower=15):
    upper = 100 - lower
    df['depanx_c'] = depanx_interaction(df, lower, upper)
    df['depadhd_c'] = depadhd_interaction(df, lower, upper)
    df['anxadhd_c'] = anxadhd_interaction(df, lower, upper)
    df['anxocd_c'] = anxocd_interaction(df, lower, upper)
    df['crysflu_c'] = crysflu_interaction(df, lower, upper)

    return df


def depanx_interaction(df, lower, upper):
    col = np.where(df['depress_D_p'] >= np.nanpercentile(df['depress_D_p'], upper),
                   np.where(df['anxdisord_D_p'] <= np.nanpercentile(df['anxdisord_D_p'], lower), 'highdep_lowanx',
                            np.where(df['anxdisord_D_p'] >= np.nanpercentile(df['anxdisord_D_p'], upper),
                                     'highdep_highanx', 'other')),
                   np.where(df['depress_D_p'] <= np.nanpercentile(df['depress_D_p'], lower),
                            np.where(df['anxdisord_D_p'] >= np.nanpercentile(df['anxdisord_D_p'], upper),
                                     'lowdep_highanx',
                                     np.where(df['anxdisord_D_p'] <= np.nanpercentile(df['anxdisord_D_p'], lower),
                                              'lowdep_lowanx', 'other')), 'other'))
    return col


def depadhd_interaction(df, lower, upper):
    col = np.where(df['depress_D_p'] >= np.nanpercentile(df['depress_D_p'], upper),
                   np.where(df['adhd_D_p'] <= np.nanpercentile(df['adhd_D_p'], lower), 'highdep_lowadhd',
                            np.where(df['adhd_D_p'] >= np.nanpercentile(df['adhd_D_p'], upper),
                                     'highdep_highadhd', 'other')),
                   np.where(df['depress_D_p'] <= np.nanpercentile(df['depress_D_p'], lower),
                            np.where(df['adhd_D_p'] >= np.nanpercentile(df['adhd_D_p'], upper), 'lowdep_highadhd',
                                     np.where(df['adhd_D_p'] <= np.nanpercentile(df['adhd_D_p'], lower),
                                              'lowdep_lowadhd', 'other')), 'other'))
    return col


def anxadhd_interaction(df, lower, upper):
    col = np.where(df['anxdisord_D_p'] >= np.nanpercentile(df['anxdisord_D_p'], upper),
                   np.where(df['adhd_D_p'] <= np.nanpercentile(df['adhd_D_p'], lower), 'highanx_lowadhd',
                            np.where(df['adhd_D_p'] >= np.nanpercentile(df['adhd_D_p'], upper),
                                     'highanx_highadhd', 'other')),
                   np.where(df['anxdisord_D_p'] <= np.nanpercentile(df['anxdisord_D_p'], lower),
                            np.where(df['adhd_D_p'] >= np.nanpercentile(df['adhd_D_p'], upper), 'lowanx_highadhd',
                                     np.where(df['adhd_D_p'] <= np.nanpercentile(df['adhd_D_p'], lower),
                                              'lowanx_lowadhd', 'other')), 'other'))
    return col


def anxocd_interaction(df, lower, upper):
    col = np.where(df['anxdisord_D_p'] >= np.nanpercentile(df['anxdisord_D_p'], upper),
                   np.where(df['ocd_D_p'] <= np.nanpercentile(df['ocd_D_p'], lower), 'highanx_lowocd',
                            np.where(df['ocd_D_p'] >= np.nanpercentile(df['ocd_D_p'], upper),
                                     'highanx_highocd', 'other')),
                   np.where(df['anxdisord_D_p'] <= np.nanpercentile(df['anxdisord_D_p'], lower),
                            np.where(df['ocd_D_p'] >= np.nanpercentile(df['ocd_D_p'], upper), 'lowanx_highocd',
                                     np.where(df['ocd_D_p'] <= np.nanpercentile(df['ocd_D_p'], lower),
                                              'lowanx_lowocd', 'other')), 'other'))
    return col


def crysflu_interaction(df, lower, upper):
    col = np.where(df['tb_cryst'] >= np.nanpercentile(df['tb_cryst'], upper),
                   np.where(df['tb_fluid'] <= np.nanpercentile(df['tb_fluid'], lower), 'highcrys_lowflu',
                            np.where(df['tb_fluid'] >= np.nanpercentile(df['tb_fluid'], upper),
                                     'highcrys_highflu', 'other')),
                   np.where(df['tb_cryst'] <= np.nanpercentile(df['tb_cryst'], lower),
                            np.where(df['tb_fluid'] >= np.nanpercentile(df['tb_fluid'], upper), 'lowcrys_highflu',
                                     np.where(df['tb_fluid'] <= np.nanpercentile(df['tb_fluid'], lower),
                                              'lowcrys_lowflu', 'other')), 'other'))
    return col
