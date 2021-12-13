# Specifications of files for inclusion and columns to be extracted and renamed

from typing import NamedTuple, List

class FileSpec(NamedTuple):
    filename: str
    nonnumericcols: List[str]
    cols: dict


CognitiveSpecs = [
    FileSpec('abcd_gdss01.txt', [], {'gdt_scr_values_safe': 'gd_safebets', 'gdt_scr_values_risky': 'gd_riskybets'}),

    FileSpec('cct01.txt', [], {'cash_choice_task': 'cct'}),

    FileSpec('abcd_tbss01.txt', [],
             {"nihtbx_picvocab_agecorrected": "tb_picvocab",
              "nihtbx_flanker_agecorrected": "tb_flanker",
              "nihtbx_list_agecorrected": "tb_list",
              "nihtbx_cardsort_agecorrected": "tb_cardsort",
              "nihtbx_pattern_agecorrected": "tb_pattern",
              "nihtbx_picture_agecorrected": "tb_picture",
              "nihtbx_reading_agecorrected": "tb_reading",
              "nihtbx_fluidcomp_agecorrected": "tb_fluid",
              "nihtbx_cryst_agecorrected": "tb_cryst",
              "nihtbx_totalcomp_agecorrected": "tb_total"}),

    FileSpec("abcd_ps01.txt", [],
             {"pea_ravlt_sd_listb_tc": "ravlt_s_total",
              "pea_ravlt_sd_listb_tr": "ravlt_s_repitition",
              "pea_ravlt_sd_listb_ti": "ravlt_s_intrusions",
              "pea_ravlt_ld_trial_vii_tc": "ravlt_l_total",
              "pea_ravlt_ld_trial_vii_tr": "ravlt_l_repitition",
              "pea_ravlt_ld_trial_vii_ti": "ravlt_l_intrusions",
              "pea_wiscv_tss": "mr_total",
              "pea_wiscv_item_a_rs": "mr_matrix",
              "pea_wiscv_item_b_rs": "mr_serial"}),

    FileSpec("abcd_mrinback02.txt", [],
             {"tfmri_nback_beh_performflag": "nb_acceptable_performance",
              "tfmri_nb_all_beh_ctotal_nt": "nb_correct_nt",
              "tfmri_nb_all_beh_ctotal_mrt": "nb_correct_mrt",
              "tfmri_nb_all_beh_c2b_nt": "nb_correct_nt_2back",
              "tfmri_nb_all_beh_c2b_mrt": "nb_correct_mrt_2back",
              "tfmri_nb_all_beh_cpf_nt": "nb_correct_nt_pos",
              "tfmri_nb_all_beh_cpf_mrt": "nb_correct_mrt_pos",
              "tfmri_nb_all_beh_cnf_nt": "nb_correct_nt_neutral",
              "tfmri_nb_all_beh_cnf_mrt": "nb_correct_mrt_neutral",
              "tfmri_nb_all_beh_cngf_nt": "nb_correct_nt_neg",
              "tfmri_nb_all_beh_cngf_mrt": "nb_correct_mrt_neg"}),

    FileSpec("mribrec02.txt", [],
             {"tfmri_rec_all_beh_posface_pr": "nb2_accuracy_pos",
              "tfmri_rec_all_beh_posface_br": "nb2_resp_bias_pos",
              "tfmri_rec_all_beh_posf_dpr": "nb2_D_prime_pos",
              "tfmri_rec_all_beh_negface_pr": "nb2_accuracy_neg",
              "tfmri_rec_all_beh_negface_br": "nb2_resp_bias_neg",
              "tfmri_rec_all_beh_negf_dp": "nb2_D_prime_neg"}),

    FileSpec("abcd_sst02.txt", [],
             {"tfmri_sst_all_beh_total_mssrt": "sst_ssrt_mean_est",
              "tfmri_sst_all_beh_total_issrt": "sst_ssrt_int_est",
              "tfmri_sst_beh_performflag": "sst_acceptable_performance"}),  # include if = 1

    FileSpec("abcd_mid02.txt", [],
             {"tfmri_mid_all_beh_srw_mrt": "mid_mrt_smrw",
              "tfmri_mid_all_beh_lrw_mrt": "mid_mrt_lgrw",
              "tfmri_mid_all_beh_t_earnings": "mid_total_payout",
              "tfmri_mid_beh_performflag": "mid_acceptable_performance",
              "tfmri_mid_all_beh_t_nt": "mid_num_trials"}),  # include if = 1

    FileSpec("lmtp201.txt", [],
             {"lmt_scr_perc_correct": "lmt_accuracy",
              "lmt_scr_num_correct": "lmt_correct_nt",
              "lmt_scr_avg_rt": "lmt_mrt",
              "lmt_scr_rt_correct": "lmt_correct_mrt",
              "lmt_scr_rt_wrong": "lmt_incorrect_mrt"}),

    FileSpec("abcd_yest01.txt", [],
             {"strp_scr_acc_all": "str_accuracy",
              "strp_scr_acc_incongr": "str_accuracy_ic",
              "strp_scr_acc_congr": "str_accuracy_c",
              "strp_scr_mnrt_all": "str_mrt",
              "strp_scr_mnrt_incongr": "str_mrt_ic",
              "strp_scr_mnrt_congr": "str_mrt_c"}),

    FileSpec("abcd_yddss01.txt", [],
             {"ddis_scr_param_delayrewamt": "ddt_rew_amnt",
              "ddis_scr_expr_mnrt_allcho": "ddt_mrt",
              "ddis_scr_expr_mnrt_delaycho": "ddt_mrt_delayed",
              "ddis_scr_expr_mnrt_immcho": "ddt_mrt_immediate",
              "ddis_scr_val_immedcho": "ddt_choice_validity"}),

    FileSpec('smarte_sumscores01.txt', [],
             {'smarte_ss_sdf_rule_rt': 'correctRT_singlearith',
              'smarte_ss_sdf_total_corr': 'num_correct_singlearith',
              'smarte_ss_mdf_challenge_acc': 'accu_mixeddigitarith',
              'smarte_ss_all_total_corr': 'totalcorrect_arith',
              'smarte_ss_dot_total_corr': 'totalcorrect_enum'})]  # exclude if < 3?

DemoSpecs = [
    # Demographic
    FileSpec('abcd_tztab01.txt', [],
             {'zyg_ss_r1_judgement': 'twin'}),

    FileSpec('abcd_ant01.txt', ['sex'],
             {'sex': 'sex', 'anthroheightcalc': 'height', 'anthroweight1lb': 'w1',
              'anthroweight2lb': 'w2', 'anthroweight3lb': 'w3',
              'anthro_waist_cm': 'waist'}),

    FileSpec("abcd_lpds01.txt", [],
             {"demo_prnt_prtnr_v2_l": "p_have_partner", "demo_prnt_prtnr_bio_l": "partner_bio_parent",
              "demo_prnt_marital_v2_l": "marital_status"}),

    FileSpec('pdem02.txt', [],
             {'demo_relig_v2': 'child_religion', 'demo_yrs_1': 'religious_service_frequency',
              'demo_yrs_2': 'relig_importance', 'demo_race_a_p___10': 'child_white',
              'demo_race_a_p___11': 'child_black', "demo_prnt_ed_v2": "parent_education",
              "demo_comb_income_v2": "parent_income", 'demo_prnt_age_v2': 'parent_age',
              "demo_fam_exp1_v2": "struggle_food_expenses", 'demo_prnt_gender_id_v2': 'sex_P'}),

    FileSpec("fhxp102.txt", [],
             {"fhx_3a_p": "#_brothers_p", "fhx_3b_p": "#_sisters_p"}),

    FileSpec('abcd_fhxssp01.txt', [],
             {"famhx_ss_moth_prob_dprs_p": "mother_dep", "famhx_ss_fath_prob_dprs_p": "father_dep",
              'famhx_ss_momdad_dprs_p': "parent_dep", "famhx_ss_momdad_scd_p": "parent_suicide",
              "famhx_ss_fath_prob_alc_p": "father_alcohol", "famhx_ss_moth_prob_alc_p": "mother_alcohol",
              "famhx_ss_fath_prob_dg_p": "father_druguse", "famhx_ss_moth_prob_dg_p": "mother_druguse",
              "famhx_ss_patgf_prob_dprs_p": "d_grandfather_dep",
              "famhx_ss_patgm_prob_dprs_p": "d_grandmother_dep",
              "famhx_ss_matgf_prob_dprs_p": "m_grandfather_dep",
              "famhx_ss_matgm_prob_dprs_p": "m_grandmother_dep",
              "famhx_ss_fulsibo1_prob_dprs_p": "sibling_dep", "famhx_ss_fath_prob_ma_p": "father_mania",
              "famhx_ss_moth_prob_ma_p": "mother_mania", "famhx_ss_fath_prob_trb_p": "father_trouble",
              "famhx_ss_parent_hspd_p": "parent_hospitalized_emo",
              "famhx_ss_parent_prf_p": "parent_therapy_emo"}),

    # Physical health
    FileSpec('abcd_hsss01.txt', [],
             {'hormone_scr_dhea_mean': 'saliva_DHEA',
              'hormone_scr_hse_mean': 'saliva_estradiol',
              'hormone_scr_ert_mean': 'saliva_testosterone'}),

    FileSpec('abcd_bp01.txt', [],
             {'blood_pressure_sys_mean': 'blood_pressure_sys',
              'blood_pressure_dia_mean': 'blood_pressure_dia'}),

    FileSpec('abcd_fbwpas01.txt', [],
             {'fit_ss_meet_abcd_rule': 'fitbit_include',
              'fit_ss_fitbit_rest_hr': 'fitbit_resting_hr',
              'fit_ss_perday_totalsteps': 'fitbit_steps',
              'fit_ss_perday_sedentarymin': 'fitbit_sedentary_mins',
              'fit_ss_perday_lightlyactivemin': 'fitbit_lightlyactive_mins',
              'fit_ss_perday_fairlyactivemin': 'fitbit_fairlyactive_mins',
              'fit_ss_perday_veryactivemin': 'fitbit_veryactive_mins'}),

    FileSpec("abcd_ysuip01.txt", [],
             {"tlfb_tob_puff_l": "tried_tabacco_k", "tlfb_mj_puff_l": "tried_weed_k",
              "tlfb_alc_sip_l": "tried_alcohol_k"}),

    FileSpec('abcd_yle01.txt', [], {'ple_ill_y': 'seriously_sick_lastyear_k'}),

    FileSpec('abcd_pq01.txt', [],
             {"pain_last_month": "pain_last_month_k", 'pain_scale_worst': 'pain_scale_worst_k'}),

    FileSpec('abcd_ssphy01.txt', ["sex"],
             {"sex": "sex", "pds_y_ss_female_category_2": 'female_puberty',
              "pds_y_ss_male_cat_2": 'male_puberty'}),

    FileSpec('abcd_lpsaiq01.txt', [],
             {'sai_p_activities_l___29': 'no_sports_activities_p'}),

    FileSpec('abcd_lpmh01.txt', [],
             {"medhx_1a_l": "medhx_doctorvisit_p",
              "medhx_2a_l": "medhx_asthma_p",
              "medhx_2b_l": "medhx_allergies_p",
              "medhx_2c_l": "medhx_brain_p",
              "medhx_2g_l": "medhx_diabetes_p",
              "medhx_2h_l": "medhx_epilepsy_p",
              "medhx_2o_l": "medhx_heart_p",
              "medhx_2q_l": "medhx_headaches_p",
              "medhx_4a_l": "medhx_emergencyroom_p",
              "medhx_6a_l": "medhx_brokenbones_p",
              "medhx_6e_l": "medhx_seriousinjury_p"}),

    # Environment

    FileSpec('abcd_saag01.txt', [], {'sag_grade_type': 'bad_grades'}),

    FileSpec('abcd_peq01.txt', [],
             {"peq_left_out_vic": "feels_leftout_k", "peq_invite_vic": "not_invited_k",
              "peq_exclude_vic": "excluded_k", "peq_rumor_vic": "otherkids_spreadneg_rumors_k",
              "peq_gossip_vic": "otherkids_gossip_k", "peq_threat_vic": "feels_threatned_k",
              "peq_loser_perp": "saysmeanthings_others_k", "peq_loser_vic": "otherkids_saymeanthings_k"}),

    FileSpec('srpf01.txt', [],
             {"school_3_y": "getalong_teachers_k", "school_6_y": "feelsafe_at_school_k",
              "school_9_y": "feels_smart_k", "school_12_y": "enjoys_school_k",
              "school_15_y": "finds_schoolboring_k", "school_17_y": "grades_important_k"}),

    FileSpec('abcd_sscey01.txt', [],
             {'pmq_y_ss_mean': 'parent_monitoring_ss_k',
              'fes_y_ss_fc_pr': 'family_conflict_ss_k',
              'psb_y_ss_mean': 'prosocial_ss_k',
              'srpf_y_ss_ses': 'school_environment_ss_k',
              'srpf_y_ss_iiss': 'school_involvement_ss_k',
              'srpf_y_ss_dfs': 'school_disengagement_ss_k',
              'wps_ss_sum': 'problem_solving_ss_k',
              'dim_y_ss_mean': 'discrimination_ss_k',
              'pbp_ss_prosocial_peers': 'peers_beh_prosocial_ss_k',
              'pbp_ss_rule_break': 'peers_beh_delinquent_ss_k',
              'pnh_ss_protective_scale': 'peer_net_protective_ss_k'}),

    FileSpec('neglectful_behavior01.txt', [],
             {'mnbs_bad' :   'parent_care_misbehave_k',
              'mnbs_doing' :   'parent_care_whereabouts_k',
              'mnbs_friend'  :  'parent_care_friends_k',
              'mnbs_homework':    'parent_helphomework_k',
              'mnbs_play' :   'parent_safeplay_k',
              'mnbs_school' :   'parent_gotoschool_k',
              'mnbs_trouble':    'parent_troubleschool_k',
              'mnbs_understand' :   'parent_helpunderstanding_k'}),

    FileSpec('abcd_siss01.txt', [],
             {'sit_scr_expr_mfinalrat': 'socialinfluence_meanfinal_k'}),

    FileSpec('abcd_ssmty01.txt', [],
             {"stq_y_ss_weekday": "screentime_weekday_ss_k", "stq_y_ss_weekend": "screentime_weekend_ss_k"}),

    FileSpec("sports_activ_read_music01.txt", [],
             {"sai_read_enjoy_y": "enjoys_reading_k", "sai_lmusic_y": "enjoys_music_k"}),

    FileSpec("abcd_ydmes01.txt", [],
             {"dim_yesno_q1": "feels_discriminated_k", "dim_matrix_q4": "senses_racism_k",
              "dim_matrix_q6": "doesnt_feel_accepted_k"}),

    FileSpec('abcd_ysr01.txt', [],
             {'resiliency5b_y': 'close_boy_friends_k', 'resiliency6b_y': 'close_girl_friends_k'}),

    FileSpec('abcd_stq01.txt', [],
             {'screentime_sq7': 'socialmedia_daysperweek_k',
              'screentime_smq_soc_med_hr': 'socialmedia_hoursperday_k',
              'screentime_sq4': 'videogames_daysperweek_k', 'screentime_smq_instagram': 'instagram_account_k'}),

    FileSpec('abcd_cb01.txt', [],
             {"cybb_phenx_harm": "bullied_on_internet_k",
              "cybb_phenx_harm_often": "bullied_internet_frequency_k"}),

    FileSpec('fes02.txt', [],
             {"fam_enviro1_p": "frequent_family_conflict_p", "fam_enviro2r_p": "family_anger_rare_p",
              "fam_enviro7r_p": "family_peaceful_p", "fes_19_p": "family_organized_p",
              "fes_27r_p": "family_nosports_p", "fes_31_p": "family_feels_togetherness_p",
              "fes_37_p": "family_activities_together_p",
              "fes_46r_p": "family_rarely_intellectualdiscourse_p",
              "fes_1_p": "family_frequent_support_p",
              "fes_71_p": "family_conflict_p", "fes_2r_p": "family_not_talk_aboutfeelings_p",
              "fes_86_p": "family_eclectic_interests_p",
              "fes_87r_p": "family_frequent_TV_p", "fes_12_p": "family_open_discussing_anything_p"}),

    FileSpec('abcd_sscep01.txt', [],
             {'nsc_p_ss_mean_3_items': 'neighborhood_safety_ss_p',
              'psb_p_ss_mean': 'prosocial_ss_p',
              'fes_p_ss_fc_pr': 'family_conflict_ss_p',
              'fes_p_ss_exp_sum_pr': 'family_expression_ss_p',
              'fes_p_ss_int_cult_sum_pr': 'family_intellectual_ss_p',
              'fes_p_ss_act_rec_sum_pr': 'family_activities_ss_p',
              'fes_p_ss_org_sum_pr': 'family_organisation_ss_p'}),

    #             FileSpec('abcd_sss01.txt', [],
    #                      {'latent_factor_ss_general_ses': 'latent_generalSES',
    #                       'latent_factor_ss_social': 'latent_socialsupport',
    #                       'latent_factor_ss_perinatal': 'latent_perinatal_health'}),

    FileSpec("abcd_lsssa01.txt", [],
             {"sai_ss_lmusic_hours_2_p_l": "enjoys_music_p"}),

    FileSpec('abcd_ssbpmtf01.txt', [],
             {"bpm_t_scr_totalprob_t": "atschool_total_problems_ss_t", "bpm_t_scr_external_t": "atschool_external_ss_t",
              "bpm_t_scr_internal_t": "atschool_internal_ss_t", "bpm_t_scr_attention_t": "atschool_attention_ss_t"}),

    # Parent mental health

    FileSpec('abcd_ksad01.txt', [],
             {"ksads_1_1_p": "parent_depressed_mood_B_p", "ksads_1_5_p": "parent_anhedonia_B_p",
              "ksads_2_7_p": "parent_elevated_mood_B_p",
              "ksads_10_45_p": "parent_excessive_worry_B_p", "ksads_16_98_p": "parent_lying_B_p",
              "ksads_23_145_p": "parent_wish_dead_present_B_p",
              "ksads_23_148_p": "parent_suicide_past_B_p", "ksads_1_160_p": "parent_fatigue_past_B_p",
              "ksads_1_159_p": "parent_fatigue_present_B_p",
              "ksads_1_179_p": "hopeless_B_p", "ksads_1_185_p": "parent_two_more_depression_B_p",
              "ksads_8_301_p": "parent_socially_anxious_B_p",
              "ksads_1_843_p": "parent_dysthymia_B_p", "ksads_8_863_p": "parent_social_anxiety_disorder_B_p",
              "ksads_22_969_p": "parent_sleep_problem_B_p",  # "ksads_13_932_p": "anorexia_B_p",
              "ksads_13_935_p": "parent_bulimia_B_p"}),

    FileSpec('abcd_asrs01.txt', [],
             {'asr_scr_anxdep_t': 'parent_anxdep_D_p',
              'asr_scr_attention_t': 'parent_attention_D_p',
              'asr_scr_aggressive_t': 'parent_aggressive_D_p',
              'asr_scr_internal_t': 'parent_internal_D_p',
              'asr_scr_external_t': 'parent_external_D_p',
              'asr_scr_depress_t': 'parent_depress_D_p',
              'asr_scr_anxdisord_t': 'parent_anxdisord_D_p',
              'asr_scr_adhd_t': 'parent_adhd_D_p',
              'asr_scr_antisocial_t': 'parent_antisocial_D_p',
              'asr_scr_hyperactive_t': 'parent_hyperactive_D_p',
              "asr_scr_somatic_t": "parent_somatic_problems_D_p",
              "asr_scr_intrusive_t": "parent_intrusive_thoughts_D_p",
              "asr_scr_avoidant_t": "parent_avoidant_person_D_p",
              "asr_scr_perstr_t": "parent_personal_strength_D_p"}),

    # Developmental
    FileSpec('dhx01.txt', [],
             {"birth_weight_lbs": "birth_weight_p", "devhx_8_prescript_med": "prescriptionmed_pregnancy_p",
              "devhx_8_cigs_per_day": "cigs_before_pregnancy_p", "devhx_8_alcohol": "alcohol_before_pregnancy_p",
              "devhx_8_marijuana": "weed_before_pregnancy_p",
              "devhx_8_coc_crack": "cocaine_before_pregnancy_p",
              "devhx_8_her_morph": "heroin_before_pregnancy_p",
              "devhx_8_other2_name_2": "drugs_before_pregnancy_p",
              "devhx_9_tobacco": "cigs_during_pregnancy_p",
              "devhx_9_alcohol": "alcohol_during_pregnancy_p",
              "devhx_9_alchohol_avg": "drinksperweek_during_pregnancy_p",
              "devhx_9_marijuana": "weed_during_pregnancy_p",
              "devhx_9_coc_crack": "cocaine_during_pregnancy_p",
              "devhx_9_her_morph": "heroin_during_pregnancy_p",
              "devhx_9_other2_name_2": "drugs_during_pregnancy_p",
              "devhx_caffeine_amt": "caffeine_during_pregnancy_p",
              "devhx_12a_p": "premature_birth_p", "devhx_18_p": "months_breastfed_p",
              "devhx_19d_p": "firstwords_months_p"})

]

NeuralDataSpecs = [FileSpec('abcd_betnet02.txt', [],
                            {"rsfmri_c_ngd_dt_ngd_dt": "rsfmri_DMN_intra",
                             "rsfmri_c_ngd_sa_ngd_sa": "rsfmri_SAN_intra",
                             "rsfmri_c_ngd_fo_ngd_fo": "rsfmri_FPN_intra",
                             "rsfmri_c_ngd_cgc_ngd_cgc": "rsfmri_CON_intra",
                             "rsfmri_c_ngd_vta_ngd_vta": "rsfmri_VAN_intra",
                             "rsfmri_c_ngd_dla_ngd_dla": "rsfmri_DAN_intra",
                             "rsfmri_c_ngd_dt_ngd_fo": "rsfmri_DMN_FPN_inter",
                             "rsfmri_c_ngd_dt_ngd_cgc": "rsfmri_DMN_CON_inter",
                             "rsfmri_c_ngd_stnvols": "rsfmri_frames_fd",
                             "rsfmri_c_ngd_stcontignvols": "rsfmri_frames_fg_contig"})]
# Child Mental Health
OutcomeSpecs = [FileSpec('abcd_ksad501.txt', [],
                         {"ksads_1_1_t": "depressed_mood_B_k", "ksads_1_5_t": "anhedonia_B_k",
                          "ksads_2_7_t": "elevated_mood_B_k",
                          "ksads_10_45_t": "excessive_worry_B_k", "ksads_16_98_t": "lying_B_k",
                          "ksads_23_145_t": "wish_dead_present_B_k",
                          "ksads_23_148_t": "suicide_past_B_k", "ksads_1_160_t": "fatigue_past_B_k",
                          "ksads_1_159_t": "fatigue_present_B_k",
                          "ksads_1_179_t": "hopeless_B_k", "ksads_1_185_t": "two_more_depression_B_k",
                          "ksads_8_301_t": "socially_anxious_B_k",
                          "ksads_8_863_t": "social_anxiety_disorder_B_k",
                          "ksads_22_969_t": "sleep_problem_B_k", "ksads_13_932_t": "anorexia_B_k",
                          "ksads_13_935_t": "bulimia_B_k"}),

                FileSpec("abcd_mhy02.txt", [],
                         {"ple_y_ss_total_good": "g_lifeevents_ss_k", "ple_y_ss_total_bad": "b_lifeevents_ss_k",
                          "ple_y_ss_affected_bad_sum": "b_lifeevents_affected_ss_k",
                          "upps_y_ss_negative_urgency": "up_negative_urgency_ss_k",
                          "upps_y_ss_lack_of_planning": "up_lackofplanning_ss_k",
                          "upps_y_ss_sensation_seeking": "up_sensationseeking_ss_k",
                          "upps_y_ss_positive_urgency": "up_positiveurgency_ss_k",
                          "upps_y_ss_lack_of_perseverance": "up_lackperseverance_ss_k",
                          "bis_y_ss_bis_sum": "bis_behav_inhibition_ss_k",
                          "bis_y_ss_bas_rr": "bis_reward_responsive_ss_k", "bis_y_ss_bas_drive": "bis_drive_ss_k",
                          "bis_y_ss_bas_fs": "bis_funseeking_ss_k",
                          "sup_y_ss_sum": "mania_7up_ss_k", "peq_ss_relational_victim": "relational_victimization_ss_k",
                          "peq_ss_reputation_aggs": "reputational_aggression_ss_k",
                          "peq_ss_reputation_victim": "reputational_victimization_ss_k",
                          "peq_ss_overt_aggression": "overt_aggression_ss_k",
                          "peq_ss_overt_victim": "overt_victimization_ss_k",
                          "peq_ss_relational_aggs": "relational_aggression_ss_k"}),

                FileSpec('emotion_reg_erq_feelings01.txt', [],
                         {'erq_feelings_control': 'emoreg_sup_control_k',
                          'erq_feelings_happy': 'emoreg_reapp_happy_k',
                          'erq_feelings_hide': 'emoreg_sup_hide_k',
                          'erq_feelings_less_bad': 'emoreg_reapp_less_bad_k',
                          'erq_feelings_self': 'emoreg_sup_self_k',
                          'erq_feelings_think': 'emoreg_reapp_thoughts_k'}),

                #             FileSpec("yabcdcovid19questionnaire01.txt", [],
                #                      { "demo_exercise_cv": "covid_exercise",
                #                       "demo_stayed_away_cv": "covid_cautious"}),
                #                       "demo_afraid_cv": "covid_afraid",
                #                       "blm_2_cv": "blm_social_post", "angry_blm_2_cv": "blm_angry", "pstr_cv_raw_tot": "covid_stress",
                #                       "nih_posaff_cv_raw_tot": "covid_happy", "nih_sad_cv_raw_tot": "covid_sad"}),

                FileSpec('abcd_cbcls01.txt', [],
                         {'cbcl_scr_syn_anxdep_t': 'anxdep_D_p',
                          'cbcl_scr_syn_attention_t': 'attention_D_p',
                          'cbcl_scr_syn_aggressive_t': 'aggressive_D_p',
                          'cbcl_scr_syn_internal_t': 'internal_D_p',
                          'cbcl_scr_syn_external_t': 'external_D_p',
                          'cbcl_scr_syn_somatic_t': 'somatic_problems_D_p',
                          'cbcl_scr_syn_social_t': 'social_problems_D_p',
                          'cbcl_scr_syn_thought_t': 'thought_disorder_D_p',
                          'cbcl_scr_syn_rulebreak_t': 'rule_breaking_D_p',
                          'cbcl_scr_dsm5_depress_t': 'depress_D_p',
                          'cbcl_scr_dsm5_anxdisord_t': 'anxdisord_D_p',
                          'cbcl_scr_dsm5_adhd_t': 'adhd_D_p',
                          'cbcl_scr_07_ocd_t': 'ocd_D_p'}),

                FileSpec("abcd_mhp02.txt", [],
                         {"ple_p_ss_total_good": "g_lifeevents_ss_p", "ple_p_ss_total_bad": "b_lifeevents_ss_p",
                          "ple_p_ss_affected_bad_sum": "b_lifeevents_affected_ss_p"}),
                #                      "eatq_p_ss_depressive_mood": "phen_depressedmood_p",
                #                      "eatq_p_ss_effort_cont_ss": "effortfulcontrol_p",
                #                      "eatq_p_ss_shyness": "shyness_p",
                #                      "eatq_p_ss_frustration": "frustration_p"}),

                FileSpec("abcd_sds01.txt", [],
                         {"sleepdisturb5_p": "difficulty_goingtosleep_p", "sleepdisturb22_p": "difficulty_wakingup_p",
                          "sleepdisturb25_p": "daytime_sleepiness_p"}),

                FileSpec("barkley_exec_func01.txt", [],
                         {"bdefs_calm_down_p": "bdefs_calm_down_p", "bdefs_consequences_p": "bdefs_consequences_p",
                          "bdefs_distract_upset_p": "bdefs_distract_upset_p",
                          "bdefs_explain_idea_p": "bdefs_explain_idea_p",
                          "bdefs_explain_pt_p": "bdefs_explain_pt_p", "bdefs_explain_seq_p": "bdefs_explain_seq_p",
                          "bdefs_impulsive_action_p": "bdefs_impulsive_action_p",
                          "bdefs_inconsistant_p": "bdefs_inconsistant_p",
                          "bdefs_lazy_p": "bdefs_lazy_p", "bdefs_process_info_p": "bdefs_process_info_p",
                          "bdefs_rechannel_p": "bdefs_rechannel_p",
                          "bdefs_sense_time_p": "bdefs_sense_time_p", "bdefs_shortcuts_p": "bdefs_shortcuts_p",
                          #                       "bdefs_start_time": "bdefs_start_time",
                          "bdefs_stop_think_p": "bdefs_stop_think_p"}),

                FileSpec('abcd_eatqp01.txt', [],
                         {'eatq_irritated_crit_p': 'easily_offended_p', 'eatq_blame_p': 'blames_others_p',
                          'eatq_social_p': 'sociable_p', 'eatq_hardly_sad_p': 'rarely_sad_p',
                          'eatq_school_excite_p': 'school_excitement_p',
                          'eatq_no_criticize_p': 'not_critical_others_p', 'eatq_puts_off_p': 'procrastination_p',
                          'eatq_friendly_p': 'friendly_p', 'eatq_dark_scared_p': 'scared_dark_p',
                          'eatq_disagree_p': 'disagreeable_p', 'eatq_stick_to_plan_p': 'goal_continuity_p'}),

                FileSpec('abcd_ple01.txt', [],
                         {'ple_died_p': 'death_in_family_p', 'ple_crime_p': 'experienced_crime_p',
                          'ple_financial_p': 'positive_finance_p', 'ple_argue_p': 'parents_argue_more_p',
                          'ple_mh_p': 'family_emotionprob_p', 'ple_separ_p': 'parents_divorced_p',
                          'ple_school_p': 'child_newschool_p', 'ple_move_p': 'family_move_p'}),

                FileSpec('abcd_cbcl01.txt', [],
                         {'cbcl_q02_p': 'druguse_alcohol_p',
                          'cbcl_q03_p': 'argues_p',
                          'cbcl_q04_p': 'doesnt_finish_p',
                          'cbcl_q05_p': 'enjoys_little_p',
                          'cbcl_q06_p': 'bad_toilet_habits_p',
                          'cbcl_q07_p': 'bragadocious_p',
                          'cbcl_q08_p': 'cant_concentrate_p',
                          'cbcl_q09_p': 'obsessions_p',
                          'cbcl_q100_p': 'sleep_problems_p',
                          'cbcl_q101_p': 'skips_school_p',
                          'cbcl_q102_p': 'low_energy_p',
                          'cbcl_q103_p': 'sad_p',
                          'cbcl_q104_p': 'loud_p',
                          'cbcl_q105_p': 'druguse_other_p',
                          'cbcl_q106_p': 'vandalism_p',
                          'cbcl_q107_p': 'wets_other_p',
                          'cbcl_q108_p': 'wets_bed_p',
                          'cbcl_q109_p': 'whines_p',
                          'cbcl_q10_p': 'hyperactive_p',
                          'cbcl_q110_p': 'wishes_other_sex_p',
                          'cbcl_q111_p': 'withdrawn_p',
                          'cbcl_q112_p': 'worries_p',
                          'cbcl_q11_p': 'clings_to_adults_p',
                          'cbcl_q12_p': 'lonely_p',
                          'cbcl_q13_p': 'confused_p',
                          'cbcl_q14_p': 'cries_p',
                          'cbcl_q15_p': 'cruel_animals_p',
                          'cbcl_q16_p': 'bullies_others_p',
                          'cbcl_q17_p': 'daydreams_p',
                          'cbcl_q18_p': 'selfharm_p',
                          'cbcl_q19_p': 'demands_attention_p',
                          'cbcl_q20_p': 'destroys_own_things_p',
                          'cbcl_q21_p': 'destroys_others_things_p',
                          'cbcl_q22_p': 'disobedient_home_p',
                          'cbcl_q23_p': 'disobedient_school_p',
                          'cbcl_q24_p': 'bad_diet_p',
                          'cbcl_q25_p': 'doesnt_get_along_p',
                          'cbcl_q26_p': 'no_guilt_p',
                          'cbcl_q27_p': 'easily_jealous_p',
                          'cbcl_q28_p': 'breaks_rules_p',
                          'cbcl_q29_p': 'fears_excl_school_p',
                          'cbcl_q30_p': 'fears_school_p',
                          'cbcl_q31_p': 'fears_being_bad_p',
                          'cbcl_q32_p': 'perfectionist_p',
                          'cbcl_q33_p': 'feels_unloved_p',
                          'cbcl_q34_p': 'paranoid_p',
                          'cbcl_q35_p': 'feels_inferior_p',
                          'cbcl_q36_p': 'accident_prone_p',
                          'cbcl_q37_p': 'fights_p',
                          'cbcl_q38_p': 'teased_p',
                          'cbcl_q39_p': 'disobedient_friends_p',
                          'cbcl_q40_p': 'hears_voices_p',
                          'cbcl_q41_p': 'impulsive_p',
                          'cbcl_q42_p': 'prefer_alone_p',
                          'cbcl_q43_p': 'lying_p',
                          'cbcl_q44_p': 'bites_fingernails_p',
                          'cbcl_q45_p': 'nervous_general_p',
                          'cbcl_q46_p': 'nervous_twitching_p',
                          'cbcl_q47_p': 'nightmares_p',
                          'cbcl_q48_p': 'not_liked_p',
                          'cbcl_q49_p': 'constipated_p',
                          'cbcl_q50_p': 'anxious_p',
                          'cbcl_q51_p': 'dizzy_p',
                          'cbcl_q52_p': 'guilty_p',
                          'cbcl_q53_p': 'overeating_p',
                          'cbcl_q54_p': 'overtired_p',
                          'cbcl_q55_p': 'overweight_p',
                          'cbcl_q56a_p': 'body_aches_p',
                          'cbcl_q56b_p': 'frequent_headaches_p',
                          'cbcl_q56c_p': 'nausea_p',
                          'cbcl_q56d_p': 'eye_problems_p',
                          'cbcl_q56e_p': 'skin_problems_p',
                          'cbcl_q56f_p': 'frequent_stomachaches_p',
                          'cbcl_q56g_p': 'vomiting_p',
                          'cbcl_q56h_p': 'unknown_physical_problems_p',
                          'cbcl_q57_p': 'attacks_others_p',
                          'cbcl_q58_p': 'picks_skin_p',
                          'cbcl_q59_p': 'plays_genitals_public_p',
                          'cbcl_q60_p': 'plays_genitals_excessive_p',
                          'cbcl_q61_p': 'poor_schoolwork_p',
                          'cbcl_q62_p': 'clumsy_p',
                          'cbcl_q63_p': 'prefers_older_kids_p',
                          'cbcl_q64_p': 'prefers_younger_kids_p',
                          'cbcl_q65_p': 'doesnt_talk_p',
                          'cbcl_q66_p': 'compulsions_p',
                          'cbcl_q67_p': 'runs_away_p',
                          'cbcl_q68_p': 'screams_p',
                          'cbcl_q69_p': 'secretive_p',
                          'cbcl_q70_p': 'hallucinations_p',
                          'cbcl_q71_p': 'easily_embarrassed_p',
                          'cbcl_q72_p': 'pyromaniac_p',
                          'cbcl_q73_p': 'sexual_problems_p',
                          'cbcl_q74_p': 'clowning_p',
                          'cbcl_q75_p': 'shy_p',
                          'cbcl_q76_p': 'sleeps_little_p',
                          'cbcl_q77_p': 'sleeps_alot_p',
                          'cbcl_q78_p': 'easily_distracted_p',
                          'cbcl_q79_p': 'speech_problems_p',
                          'cbcl_q80_p': 'blank_stare_p',
                          'cbcl_q81_p': 'steals_home_p',
                          'cbcl_q82_p': 'steals_outside_p',
                          'cbcl_q83_p': 'hordes_p',
                          'cbcl_q84_p': 'strange_ideas_p',
                          'cbcl_q85_p': 'strange_behavior_p',
                          'cbcl_q86_p': 'stubborn_p',
                          'cbcl_q87_p': 'mood_fluctuations_p',
                          'cbcl_q88_p': 'sulks_p',
                          'cbcl_q89_p': 'suspicious_p',
                          'cbcl_q90_p': 'obscene_language_p',
                          'cbcl_q91_p': 'suicidal_p',
                          'cbcl_q92_p': 'sleepwalks_p',
                          'cbcl_q93_p': 'loquacious_p',
                          'cbcl_q94_p': 'teases_p',
                          'cbcl_q95_p': 'temper_tantrums_p',
                          'cbcl_q96_p': 'sexual_thoughts_p',
                          'cbcl_q97_p': 'threatens_others_p',
                          'cbcl_q98_p': 'thumbsucking_p',
                          'cbcl_q99_p': 'druguse_tobacco'}),

                FileSpec('abcd_pssrs01.txt', [],
                         {'ssrs_16_p': 'avoids_eyecontact_p', 'ssrs_18_p': 'difficulty_making_friends_p',
                          'ssrs_29_p': 'regarded_weird_p', 'ssrs_35_p': 'bad_conversational_flow_p',
                          'ssrs_39_p': 'narrow_interests_p',
                          'ssrs_42_p': 'sensory_sensitivity_p', 'ssrs_58_p': 'concentration_on_parts_p'})]