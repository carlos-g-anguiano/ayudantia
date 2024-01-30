#!/eos/home-h/hcrottel/new_env/bin/python
import pandas as pd
import numpy as np
import json
import sys
import pickle
import os
from shutil import copyfile
import tools
#import invariantMass_tools as invMass
from termcolor import colored, cprint
import pdb
import xgboost as xgb
#import uproot3 as uproot

def print_log(string, log, color=None):
    if log: log += string
    print(colored(string, color))
    
    
def cands_per_event(df):
    try:
        return df.reset_index('subentry').index.value_counts()
    except Exception as e:
        #print(e)
        return df.index.value_counts()
    
    
def read_cuts_json(cuts_json):
    path = tools.analysis_path('scripts/cuts')
    with open(f'{path}/cuts-{cuts_json}.json', 'r') as file_:
        cuts = json.load(file_)
    return cuts


def apply_loose_cuts(df, log=None, **kwargs):
    """Loose cuts as a first approach to look for the B+ resonance
    - Bpt = 5.35,
    - kpt = 1.2,
    - prob = 0.083,
    - signLxy = 4.45,
    - cosA =0.9987,
    - PDL = 0.012
    """
    loose_cuts = dict(Bpt = 5.35,
                      kpt = 1.2,
                      prob = 0.083,
                      signLxy = 4.45,
                      cosA =0.9987,
                      PDL = 0.012
                     )
    
    for k in loose_cuts:
        if k in kwargs:
            loose_cuts[k] = kwargs[k]
    
    return apply_cuts(loose_cuts.keys(), loose_cuts, df)


def apply_quality_cuts(df, log=None, verbose=True):
    
    k_cuts  = (df.k_HighPurity==1) & (df.k_numberOfHits>=5) & (df.k_numberOfPixelHits>=1)
    mu_cuts = df.mu1_isSoft + df.mu2_isSoft==2
    mask = k_cuts & mu_cuts
    
    if verbose:
        print_log(f'\nQuality Cuts\n - k_HighPurity\n - k_numberOfHits>=5\n - k_numberOfPixelHits>=1\n - Soft Muons',
              log, color='green')
    try:
        T = len(np.unique(df[['run', 'event', 'luminosityBlock']], axis=0))
    except KeyError:
        T = len(cands_per_event(df))
        
    to_log = list()
    df = df[mask]
    try:
        T2, counts = np.unique(df[['run', 'event', 'luminosityBlock']], axis=0, return_counts=True)
        T2 = len(T2)
    except KeyError:
        cands__ = cands_per_event(df)
        T2 = len(cands__.index)
        counts = cands__.values
    
    
    to_log.append(f'Events after QC             : {T2} - {round(100*T2/T,4)}%')
    to_log.append(f'Len                         : {len(df)}')
    to_log.append(f'candidates/event (mean)     : {round(np.mean(counts), 3)}')
    if 'GENCand' in df:
        TGEN = len(df[df.GENCand])
    else:
        TGEN = 0
    to_log.append(f'     GEN len                     : {TGEN}')
    to_log.append(f'     GEN  %                      : {100*TGEN/len(df)}')
    
    if verbose:
        print_log('\n'.join(to_log), log)
    return df

    
def resonance_rejection(cuts, df, log=None):
    """Apply resonance rejection to df
    from the values defined in cuts and 
    update the log"""
    if type(cuts)==int:
        cuts = read_cuts_json(cuts)
    print_log('\nResonance Rejection', log, color='green')
    
    JPsiLow, JPsiHigh = cuts['Resonances']["JPsi"]
    resonance_JPsi = (df.DiMuMass<JPsiLow) | (df.DiMuMass>JPsiHigh)
    print_log(f'JPsi       {JPsiLow} - {JPsiHigh}', log, color='yellow')

    PsiPLow, PsiPHigh = cuts['Resonances']["PsiP"]
    resonance_PsiP = (df.DiMuMass<PsiPLow) | (df.DiMuMass>PsiPHigh)
    print_log(f'PsiPrime   {PsiPLow} - {PsiPHigh}', log, color='yellow')
    
    #T =len(df)
    try:
        T = len(np.unique(df[['run', 'event', 'luminosityBlock']], axis=0))
        df = df[resonance_JPsi & resonance_PsiP]
        T2, counts = np.unique(df[['run', 'event', 'luminosityBlock']], axis=0, return_counts=True)
        T2 = len(T2)
    except KeyError:
        T = len(cands_per_event(df))
        df = df[resonance_JPsi & resonance_PsiP]
        cands__ = cands_per_event(df)
        T2 = len(cands__.index)
        counts = cands__.values
    #to_log = [f'Events after Resonance Rejection : {len(df)} - {round(100*len(df)/T,4)}% (Del Inicial)']
    to_log = [f'Events after Resonance Rejection : {T2} - {round(100*T2/T,4)}%']
    to_log.append(f'Len                              : {len(df)}')
    to_log.append(f'candidates/event (mean)          : {round(np.mean(counts), 3)}')
    if 'GENCand' in df:
        TGEN = len(df[df.GENCand])
    else:
        TGEN = 0
    to_log.append(f'     GEN len                     : {TGEN}')
    to_log.append(f'     GEN  %                      : {100*TGEN/len(df)}')
    
    print_log('\n'.join(to_log), log)
    return df

def anti_radiation_veto(cuts, df, log=None):
    """Apply anti-radiation to df
    from the values defined in cuts and 
    update the log"""
    print_log('\nAnti-Radiation Veto', log, color='green')
    to_log=list()
    
    MB    = cuts['Constants']['Mass']['B+']
    MJPsi = cuts['Constants']['Mass']['JPsi']
    MPsiP = cuts['Constants']['Mass']['PsiP'] 
    print_log(f'\tm(B)    = {MB}\n\tm(JPsi) = {MJPsi}\n\tm(PsiP) = {MPsiP}', log, 'yellow')
    #-----------------------------------------------------------------------------------------#
    JPsi_m1 = cuts["Anti-Rad"]["JPsi"]["m1"]
    JPsi_m2 = cuts["Anti-Rad"]["JPsi"]["m2"]
    JPsi_c  = cuts["Anti-Rad"]["JPsi"]["c"]
    JPsiDif = np.abs((df.BMass - MB) - (df.DiMuMass - MJPsi)) 

    antiRad_JPsi1 = (JPsiDif > JPsi_m1) | (df.DiMuMass > MJPsi)
    antiRad_JPsi2 = (JPsiDif > JPsi_m2) | \
                    (df.DiMuMass > JPsi_c)  | \
                    (df.DiMuMass< MJPsi)
    #-----------------------------------------------------------------------------------------#
    PsiP_m1 = cuts["Anti-Rad"]["PsiP"]["m1"]
    PsiP_m2 = cuts["Anti-Rad"]["PsiP"]["m2"]
    PsiP_c  = cuts["Anti-Rad"]["PsiP"]["c"]
    PsiPDif = np.abs((df.BMass - MB) - (df.DiMuMass - MPsiP)) 

    print_log(f'JPsi --->   m1 = {JPsi_m1}   m2 = {JPsi_m2}   c = {JPsi_c}', log, color='yellow')
    print_log(f'PsiP --->   m1 = {PsiP_m1}   m2 = {PsiP_m2}   c = {PsiP_c}', log, color='yellow')

    antiRad_PsiP1 = (PsiPDif > PsiP_m1) | (df.DiMuMass > MPsiP)
    antiRad_PsiP2 = (PsiPDif > PsiP_m2) | \
                    (df.DiMuMass > PsiP_c)  | \
                    (df.DiMuMass < MPsiP)
    
    ############################### ANTI-RADIATION VETO #######################################
    try:
        T  = len(np.unique(df[['run', 'event', 'luminosityBlock']], axis=0))
        df = df[antiRad_JPsi1 & antiRad_JPsi2 & antiRad_PsiP1 & antiRad_PsiP2]
        T2, counts = np.unique(df[['run', 'event', 'luminosityBlock']], axis=0, return_counts=True)
        T2 = len(T2)
    except KeyError:
        T = len(cands_per_event(df))
        df = df[antiRad_JPsi1 & antiRad_JPsi2 & antiRad_PsiP1 & antiRad_PsiP2]
        cands__ = cands_per_event(df)
        T2 = len(cands__.index)
        counts = cands__.values
        
        
    to_log.append(f'Events after RR and Anti-Radi    : {T2} - {round(100*T2/T,4)}%')
    to_log.append(f'Len                              : {len(df)}')
    to_log.append(f'candidates/event (mean)          : {round(np.mean(counts), 3)}')
    if 'GENCand' in df:
        TGEN = len(df[df.GENCand])
    else:
        TGEN = 0
    to_log.append(f'     GEN len                     : {TGEN}')
    to_log.append(f'     GEN  %                      : {100*TGEN/len(df)}')
    
    print_log('\n'.join(to_log), log)
    return df
    
    
    

    
def produce_XGB_col(cuts, df, log=None, appendix="", verbose=True):
    if verbose:
        print('...producing XGB column\n')
    to_log=list()
    #T = len(df)
    #try:
    #    T = len(np.unique(df[['run', 'event', 'luminosityBlock']], axis=0))
    #except Exception as e:
    #    print('No `run`, `event`, `luminosityBlock` information')
    ##################################### XGBOOST #############################################
    ##################################### XGBOOST #############################################
    #Default Model
    models_dir = tools.analysis_path('XGB/Models/Oct20/')
    
    if "pathR" in cuts['XGB'] and 'xgb' in str(type(cuts["XGB"]["pathR"])).lower():
        if verbose: print('---->>>> HERE! -- R')
        right_model = cuts["XGB"]["pathR"]
    
    elif "pathR" in cuts['XGB']:
        if verbose: cprint(f'Right Model  :  {cuts["XGB"]["pathR"]}', 'yellow')
        #log+=f"Right Model  :  {cuts['XGB']['pathR']}\n"
        to_log.append('Right Model  :  '+ cuts['XGB']["pathR"])
        
        if cuts['XGB']["pathR"].endswith('.pkl'):
            right_model = pickle.load(open(tools.analysis_path(cuts['XGB']["pathR"]), 'rb')) 
        else:
            right_model = xgb.XGBClassifier()                   
            right_model.load_model(tools.analysis_path(cuts['XGB']["pathR"]))
    else:
        if verbose: cprint(f'Right Model  :  {models_dir}model2_rightSB_2.pickle.dat', 'yellow')
        #log+=f"Right Model  :  {models_dir+'model2_rightSB_2.pickle.dat'}\n"
        to_log.append('Right Model  :  '+ models_dir+'model2_rightSB_2.pickle.dat')
        try:
            right_model = pickle.load(open(models_dir+'model2_rightSB_2.pickle.dat', 'rb')) 
        except pickle.UnpicklingError as e:
            cprint(f'Cannot import Default model. Maybe due to missmatch versions of pickle:', 'magenta')
            cprint(e, 'red')
            
        
    if "pathL" in cuts["XGB"] and 'xgb' in str(type(cuts["XGB"]["pathL"])).lower():
        if verbose: print('---->>>> HERE! -- L')
        left_model = cuts["XGB"]["pathL"]
        
    elif "pathL" in cuts['XGB']:
        if verbose: cprint(f'Left Model   :  {cuts["XGB"]["pathL"]}', 'yellow')
        #log+=f"Left Model  :   {cuts['XGB']['pathL']}\n"
        to_log.append('Left Model  :  '+ cuts['XGB']["pathL"])
        if cuts['XGB']["pathL"].endswith('.pkl'):
            left_model = pickle.load(open(tools.analysis_path(cuts['XGB']["pathL"]), 'rb')) 
        else:
            left_model = xgb.XGBClassifier()                   
            left_model.load_model(tools.analysis_path(cuts['XGB']["pathL"]))                    

    else:
        if verbose:  cprint(f'Left Model   :   {models_dir}model2_leftSB_2.pickle.dat', 'yellow')
        #log+=f"Left Model  :   {models_dir+'model2_leftSB_2.pickle.dat'}\n"
        to_log.append('Left Model  :  '+ models_dir+'model2_leftSB_2.pickle.dat')
        left_model  = pickle.load(open(models_dir+'model2_leftSB_2.pickle.dat', 'rb'))

           
            
    
    cols =['Bpt', 'kpt', 'PDL', 'prob', 'cosA', 'signLxy']
    #pdb.set_trace()
    #if appendix is not None:
    temp = df.rename(
							columns = {f'cosA{appendix}': 'cosA', 
							           f'signLxy{appendix}': 'signLxy'}, 
							inplace = False
							)
    temp = temp[cols]
    #else:
    #    temp = df[cols]
        
    left_cut  = cuts['XGB'].get('Left', '')
    right_cut = cuts['XGB'].get('Right','')




    if "pathR" in cuts['XGB'] or "pathL" in cuts['XGB']:    
        if left_cut:
            df[f'L_XGB{appendix}'] = left_model.predict_proba(temp)[:,1]
        if right_cut:
            df[f'R_XGB{appendix}'] = right_model.predict_proba(temp)[:,1]
    else:
        # FOR SOME WIERD REASON, THESE LABELS WERE SWAPPED
        # LEAVE IT AS IT IS FOR CONSISNTENCY WITH VERY OLD CODE
        df[f'R_XGB{appendix}'] = left_model.predict_proba(df[cols])[:,1]
        df[f'L_XGB{appendix}'] = right_model.predict_proba(df[cols])[:,1]
        
        
    return df


    

def apply_XGB(cuts, df, log=None, appendix=""):

    print_log('\nXGBoost classifier', log, 'green')
    #Used when several PV where used. Now to be deprecated
    if appendix=='best':
        df = produce_XGB_col(cuts, df, log, 0)
        df = produce_XGB_col(cuts, df, log, 1)
        df = produce_XGB_col(cuts, df, log, 2)
        df = produce_XGB_col(cuts, df, log, 3)
    else:
        df = produce_XGB_col(cuts, df, log, appendix)

    to_log=list()
    #T = len(df)
    try:
        T  = len(np.unique(df[['run', 'event', 'luminosityBlock']], axis=0))
    except KeyError:
        T = len(cands_per_event(df))


    left_cut  = cuts['XGB'].get('Left', '')
    right_cut = cuts['XGB'].get('Right','')
    if appendix == 'best':
        if left_cut:
            df['L_XGBbest'] = np.max(df[[f'L_XGB{i}' for i in range(4)]], axis=1)
        if right_cut:
            df['R_XGBbest'] = np.max(df[[f'R_XGB{i}' for i in range(4)]], axis=1)

    """
		if "pathR" in cuts['XGB'] or "pathL" in cuts['XGB']:    
        if left_cut:
            df['L_XGB'] = left_model.predict_proba(df[cols])[:,1]
        if right_cut:
            df['R_XGB'] = right_model.predict_proba(df[cols])[:,1]
    else:
        # FOR SOME WIERD REASON, THESE LABELS ARE SWAPPED!!
        # LEAVE IT AS IT IS FOR CONSISNTENCY WITH OLD CODE!
        df['R_XGB'] = left_model.predict_proba(df[cols])[:,1]
        df['L_XGB'] = right_model.predict_proba(df[cols])[:,1]
    """
		##################################### XGBOOST 
    #############################################
    if left_cut and right_cut:
        print_log(f'LEFT AND RIGHT CUTS  --  Left = {left_cut} -  Right = {right_cut}', log, 'cyan')
        df = df [(df[f'R_XGB{appendix}']>=right_cut) & (df[f'L_XGB{appendix}']>=left_cut)]
    elif left_cut:
        print_log(f'ONLY LEFT CUT  --  Left = {left_cut}', log, 'cyan')
        df = df [(df[f'L_XGB{appendix}']>=left_cut)]
    elif right_cut:
        print_log(f'ONLY RIGHT CUT --  Right = {right_cut}', log, 'cyan')
        df = df [(df[f'R_XGB{appendix}']>=right_cut)]
    else:
        raise NotImplementedError('NO XGBOOST CUT')

        
    try:
        T2, counts = np.unique(df[['run', 'event', 'luminosityBlock']], axis=0, return_counts=True)
        T2 = len(T2)
    except KeyError:
        cands__ = cands_per_event(df)
        T2 = len(cands__.index)
        counts = cands__.values
        
    #T2, counts =np.unique(df[['run', 'event', 'luminosityBlock']], axis=0, return_counts=True)
    #T2 = len(T2)
    #to_log.append(f'Events after XGB                 : {len(df)} - {round(100*len(df)/T,4)}%')
    to_log.append(f'Events after XGB                 : {T2} - {round(100*T2/T,4)}%')
    to_log.append(f'Len                              : {len(df)}')
    to_log.append(f'candidates/event (mean)          : {round(np.mean(counts), 3)}')
    
    if 'GENCand' in df:
        TGEN = len(df[df.GENCand])
    else:
        TGEN = 0

    to_log.append(f'     GEN len                     : {TGEN}')
    if len(df)>0:
        to_log.append(f'     GEN  %                      : {100*TGEN/len(df)}\n')

    print_log('\n'.join(to_log), log)
    return df
    
    
    
    
    
def apply_missID(cuts, df, log=None):
    """Apply miss K-MU miss ID, by a cut on track-muon invariant Mass"""
    print_log('\nMiss ID cut', log, 'green')
    #df_ = invMass.calculate_invMassKmu_missID(df)
    df['InvMassMissID'] = invMass.calculate_invMassKmu_missID(df)
    #del df_
    #T  = len(df)
    try:
        T = len(np.unique(df[['run', 'event', 'luminosityBlock']], axis=0))
    except KeyError:
        T = len(cands_per_event(df))
    to_log = list()
    if 'Diagonal' in cuts:
        window = cuts['Diagonal']['Window'] 
        kpt    = cuts['Diagonal']['kpT']
        print_log(f'MissID Invariant Mass : {window[0]} - {window[1]} ', log, 'yellow')
        print_log(f'       Kpt cut        : {kpt}', log, 'yellow')
    else:
        to_log.append('WARNING!!!!     ---    NOT DIAGONAL CUT INFO IN JSON')
        raise NotImplementedError
        
    diag_cut = np.bitwise_not((df.InvMassMissID>window[0])\
                              & (df.InvMassMissID < window[1])\
                              & (df.kpt<kpt))
    
    df = df[diag_cut]
    try:
        T2, counts = np.unique(df[['run', 'event', 'luminosityBlock']], axis=0, return_counts=True)
        T2 = len(T2)
    except KeyError:
        cands__ = cands_per_event(df)
        T2 = len(cands__.index)
        counts = cands__.values
    to_log = []
    
    

    #to_log.append(f'Events after MissID veto         : {len(df)} - {round(100*len(df)/T,4)}% (Del Inicial)')
    to_log.append(f'Events after MissID veto         : {T2} - {round(100*T2/T,4)}%')
    to_log.append(f'Len                              : {len(df)}')
    to_log.append(f'candidates/event (mean)          : {round(np.mean(counts), 3)}')
    if 'GENCand' in df:
        TGEN = len(df[df.GENCand])
    else:
        TGEN = 0
    to_log.append(f'     GEN len                     : {TGEN}')
    to_log.append(f'     GEN  %                      : {100*TGEN/len(df)}')
    
    print_log('\n'.join(to_log), log)
    return df


    


def apply_simple_cut(cut, df, column, type_='g', log=None):
    """Apply simple cut on single column, 
       types:
           - g  : greater
           - ge : greater equal
           - l  : less
           - le : less equal
           - eq : equal
    """
    if   type_=='g' :   compare, mask = '>', df[column]>cut        
    elif type_=='l' :   compare, mask = '<', df[column]<cut        
    elif type_=='ge':   compare, mask = '>=', df[column]>=cut        
    elif type_=='le':   compare, mask = '<=', df[column]<=cut    
    elif type_=='eq':   compare, mask = '==', df[column]==cut        
    else:
        raise NotImplementedError(f'I do not recognize type : {type_}\n Allowed are: g, ge, l, le, eq')

    print_log(f'\nCut {column} {compare} {cut}', log, color='green')
    try:
        T = len(np.unique(df[['run', 'event', 'luminosityBlock']], axis=0))
    except KeyError:
        T = len(cands_per_event(df))
        
    to_log = list()
    df = df[mask]
    try:
        T2, counts = np.unique(df[['run', 'event', 'luminosityBlock']], axis=0, return_counts=True)
        T2 = len(T2)
    except KeyError:
        cands__ = cands_per_event(df)
        T2 = len(cands__.index)
        counts = cands__.values
    
    ##LOG
    string_ = f'Events after {column} {compare} {cut}'
    string_ += ' '* (np.max([33-len(string_), 1]))
    to_log.append(string_ +                        f': {T2} - {round(100*T2/T,4)}%')
    to_log.append(f'Len                              : {len(df)}')
    to_log.append(f'candidates/event (mean)          : {round(np.mean(counts), 3)}')
    if 'GENCand' in df:
        TGEN = len(df[df.GENCand])
    else:
        TGEN = 0
    to_log.append(f'     GEN len                     : {TGEN}')
    if len(df)>0:
        to_log.append(f'     GEN  %                      : {100*TGEN/len(df)}\n')

    print_log('\n'.join(to_log), log)
    return df
    
    

    
    
    
#Functions needed
def apply_cut(name, cuts, df):
    """
    
    Posible names:
        - resonance_rejection
        - anti_radiation_veto
        - XGB
        - missID
        - Bpt
        - mu1_pt
        - mu2_pt
        - kpt
        - Quality
        
    cuts : json file containing the 
           values for the cuts
    """
    resonances = ['resonance_rejection', 'Resonances']
    antis = ['anti_radiation_veto', 'Anti-Rad']
    misses = ['missID', 'Diagonal']
    simples = ['Bpt', 'mu1_pt', 'mu2_pt', 'kpt', 'l1pt', 'l2pt', 'k_min_dr_trk_muon']
    simples+= ['cosA', 'PDL', 'prob', 'signLxy']
    
    if name in resonances:
        df = resonance_rejection(cuts, df)
        
    elif name in antis:
        df = anti_radiation_veto(cuts, df)
        
    elif name == 'XGB':
        df = apply_XGB(cuts, df)
        
    elif name == 'missID':
        df = apply_missID(cuts, df)
        
    #elif name == 'Bpt':
    #    df = df[df.Bpt>cuts['Bpt']]
    elif name in simples:
        cut_val = cuts[name]
        df = apply_simple_cut(cut_val, df, name, type_='g')
        
    elif name=='Quality':
        df = apply_quality_cuts(df)
        
    elif name=='Constants':
        return df
    
    else:
        string = f'I do not recognize:{name}\n\n'
        string += 'Posible names:\n'\
        '  - resonance_rejection\n'\
        '  - anti_radiation_veto\n'\
        '  - XGB\n'\
        '  - missID\n'\
        '  - Bpt\n'
        raise NotImplementedError(string)
        
    return df


def apply_cuts(names, cuts, df):
    """
    Posible names:
        - resonance_rejection
        - anti_radiation_veto
        - XGB
        - missID
        - Bpt
        - Quality
        
    cuts : json file containing the 
           values for the cuts
    """
    for name in names:
        df = apply_cut(name, cuts, df)
    return df


def read_scale_factors(name='Test_All_CH_SF.root'):
    path = tools.analysis_path(f'scripts/scalefactors/{name}')
    print(f'Reading file: scalefactors/{name}')
    file = uproot.open(path)
    TH3D = file['hSF_All']
    
    ScaleFactors = TH3D.allnumpy()[0]
    ScaleFactors = ScaleFactors[:,:,1]
    Variances    = TH3D.allvariances[:,:,1] 
    Bins_x = TH3D.allnumpy()[1][0][0]
    Bins_y = TH3D.allnumpy()[1][0][1]
    Bins_z = TH3D.allnumpy()[1][0][2]
    
    return dict(ScaleFactors=ScaleFactors,
                Variances=Variances,
                Bins_x=Bins_x,
                Bins_y=Bins_y,
                Bins_z=Bins_z)
    
    
def apply_scale_factors(pT, IP, scale_dict=None, returnVariance=False):
    
    if not scale_dict:
        scale_dict = read_scale_factors()
        
    Pt_index = np.argwhere(scale_dict['Bins_x']<=pT).flatten()[-1]
    IP_index = np.argwhere(scale_dict['Bins_y']<=IP).flatten()[-1]

    if returnVariance:
        return scale_dict['ScaleFactors'][Pt_index][IP_index], scale_dict['Variances'][Pt_index][IP_index]
    return scale_dict['ScaleFactors'][Pt_index][IP_index]


def apply_SF(df):
    
    #Read ROOT Histogram and parse it into a dictionary
    SF_dict = read_scale_factors()
    #Initialize SF at zeros
    SF = np.zeros(len(df), dtype=float)

    #Iterate over all rows from DataFrame
    for index in range(len(df)):
        event_info = df.iloc[index]
        #Get muons pTs
        pT1, pT2 = event_info.l1pt, event_info.l2pt
        
        #Check if there is one triggering muon, and which one is:
        if event_info.Mu1_isTriggering==1 and event_info.Mu2_isTriggering==0:
            pT    = pT1
            IP_sig= np.abs(event_info.Mu1_IP_BeamSpot/event_info.Mu1_IPerr_BeamSpot)
        elif event_info.Mu1_isTriggering==0 and event_info.Mu2_isTriggering==1:
            pT    = pT2
            IP_sig= np.abs(event_info.Mu2_IP_BeamSpot/event_info.Mu2_IPerr_BeamSpot)
            
        #If there are two triggering muons, select the one with the lowest pT    
        elif event_info.Mu1_isTriggering==1 and event_info.Mu2_isTriggering==1:
            if pT1<pT2:
                pT    = pT1
                IP_sig= np.abs(event_info.Mu1_IP_BeamSpot/event_info.Mu1_IPerr_BeamSpot)
            else:
                pT    = pT2
                IP_sig= np.abs(event_info.Mu2_IP_BeamSpot/event_info.Mu2_IPerr_BeamSpot)
                
        #Select the SF based on the pT and IPsig only
        SF[index] = apply_scale_factors(pT, IP_sig, scale_dict=SF_dict)
        
    df['SF'] = SF
    return df

    
    
if __name__ == '__main__':
    
    import argparse
    parser    = argparse.ArgumentParser(description='Apply offline cuts from a single cand per event dataframe')
    
    type_data = parser.add_mutually_exclusive_group(required=True)
    type_data.add_argument('--MC', 
                            action='store_true',
                            help  ='Set if MonteCarlo Dataset')
    type_data.add_argument('--RD',
                            action='store_true', 
                            help='Set if Real Data (Collision Data)')
	
    parser.add_argument('-c',
						action='store',
						default=None, 
                       help='ID for cuts.json')
    parser.add_argument('--inputfile',
                       action='store',
                       required=False, 
                       help='Path to read the pd.DataFrame')
    parser.add_argument('--inputdir',
                       action='store',
                       required=False, 
                       help='Dir to read the pd.DataFrames')
    parser.add_argument('--outputfile',
                       action='store',
                       required=False, 
                       type=str,
                       default='-',
                       help='Parent directory to save the pkl, \
                        defaults to ./Skim{c}/RR_ARV_XGB.pkl')
    parser.add_argument('--appendix',
                        action='store',
                        required=False,
                        type=str,
                        default='',
                        help='Which PV to use, defaults to "" which means that the primary vertex has been already selected (e.g. in the previous step)')
    
    args = parser.parse_args() 
    
    
    
    
    if args.inputfile:
        log  = ''
        cuts = read_cuts_json(args.c)
        data = pd.read_pickle(args.inputfile)
    
    
    
    
        print('\n') 
        data = resonance_rejection(cuts, data, log)
        print('\n')
        data = anti_radiation_veto(cuts, data, log)
        print('\n')
        data = apply_XGB(cuts, data, log, args.appendix)
        print('\n')
        #data = apply_missID(cuts, data, log)
        #print('\n')


        out_file = ''
        if args.outputfile=='-':
            out_file = f'./Skim{args.c}/RR_ARV_XGB.pkl'
        else:
            out_file = args.outputfile
        
        if len(out_file.split('/'))>1:
            path_to = os.path.join(*out_file.split('/')[:-1])
        else:
            path_to = '.'
        os.makedirs(path_to, exist_ok=True)
        pd.to_pickle(data, out_file)


        with open(os.path.join(path_to, 'skim.log'), 'w+') as ll:
            ll.write(log)    

            
            
            
            
            
    elif args.inputdir:
        log  = ''
        cuts = read_cuts_json(args.c)
        
        
        if args.MC:
            data = pd.DataFrame()
            for f in os.listdir(args.inputdir):
                if not f.endswith('.pkl'): continuepycms
                temp = pd.read_pickle(os.path.join(args.inputdir, f))
                data = data.append(temp, ignore_index=False)
            print('\n') 
            data = resonance_rejection(cuts, data, log)
            print('\n')
            data = anti_radiation_veto(cuts, data, log)
            print('\n')
            data = apply_XGB(cuts, data, log, args.appendix)
            print('\n')
            #data = apply_missID(cuts, data, log)
            #print('\n')

            out_file = ''
            if args.outputfile=='-':
                out_file = f'./Skim{args.c}/RR_ARV_XGB.pkl'
            else:
                out_file = args.outputfile

            path_to = os.path.join(*out_file.split('/')[:-1])
            os.makedirs(path_to, exist_ok=True)
            pd.to_pickle(data, out_file)

            with open(os.path.join(path_to, 'skim.log'), 'w+') as ll:
                ll.write(log)    
                
                
                
        elif args.RD:
            for f in os.listdir(args.inputdir):
                dataset_name = f.replace('BestCand', '')
                if not f.endswith('.pkl'): continue
                data =pd.read_pickle(os.path.join(args.inputdir, f))
                print('\n') 
                data = resonance_rejection(cuts, data, log)
                print('\n')
                data = anti_radiation_veto(cuts, data, log)
                print('\n')
                data = apply_XGB(cuts, data, log, args.appendix)
                print('\n')
                data = apply_missID(cuts, data, log)
                print('\n')

                out_file = ''
                if args.outputfile=='-':
                    out_file = f'./Skim{args.c}/RR_ARV_XGB{dataset_name}'
                else:
                    out_file = args.outputfile

                path_to = os.path.join(*out_file.split('/')[:-1])
                os.makedirs(path_to, exist_ok=True)
                pd.to_pickle(data, out_file)

                with open(os.path.join(path_to, 'skim.log'), 'w+') as ll:
                    ll.write(log) 