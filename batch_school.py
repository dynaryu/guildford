import pandas as pd
import os
import shapefile
#from shapely.geometry import Polygon
#from shapely.geometry import Point
import numpy as np
from scipy import stats
import cPickle

from eqrm_code.worden_et_al import worden_et_al
from eqrm_code.RSA2MMI import rsa2mmi_array

from batch_residential import read_hazus_casualty_data, read_hazus_collapse_rate, \
    assign_casualty, assign_damage_state, sample_vulnerability, compute_vulnerability

"""
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

import pylab
pylab.rcParams['figure.figsize'] = (8.0, 6.0)
pylab.rcParams['figure.dpi'] = 300
pylab.rcParams['font.size'] = 12
pylab.rcParams['legend.numpoints'] = 1
"""

# In[75]:
# In[78]:

def assign_casualty(df_damage, casualty_rate):
    """
    assign casualty rate for each building by building type
    """
    casualty = {}

    for severity in casualty_rate.minor_axis:

        df_casualty = pd.DataFrame(df_damage.values)

        for bldg, group in df_casualty.groupby(df_casualty.shape[1]-1):

            # print "%s: %s" % (severity, bldg)
            # replace_dic = casualty_rate[severity][bldg]
            for ids, ds in enumerate(casualty_rate.items, start=1):
                df_casualty[group == ids] = casualty_rate.loc[ds, bldg, severity]

                #df_casualty.loc[group==ds] = replace_dic[ds]
                #df_casualty.where(group==ds, replace_dic[ds], inplace=True)

        casualty[severity] = df_casualty.copy()

    return casualty

def assign_vulnerability_class(df, mapping_dic):

    try:
        bldg_ = mapping_dic[df['GA code']]
    except KeyError:
        bldg_ = df['STRUCTURE_CLASSIFICATION']

    if df['PRE1989']:
        tail = 'Pre'
    else:
        tail = 'Post'

    return pd.Series({'VUL_CLASS': '{}_{}'.format(bldg_, tail)})


# # In[4]:

# def compute_vulnerability(mmi, bldg_class):
#     gar_mmi = np.arange(4.0, 11.0, 0.05)
#     return np.interp(mmi, gar_mmi, vul_function[bldg_class])

# In[9]:


# In[25]:


def main(pdir, project_tag, site_tag, gm_tag, site_csv_file, path_vul, hazus_data_path, output_path, nsample=10):

    gm_path = os.path.join(pdir, project_tag, gm_tag)

    # Read Non-Residential Bldgs
    site = pd.read_csv(site_csv_file)
    #fields_type = [x[1] for x in sf.fields[1:]]

    # read vulnerability for non residential bldgs
    gar_mmi = np.arange(4.0, 11.0, 0.05)
    vul_function = cPickle.load(open(os.path.join(path_vul, 'vul_function_non_res.p'),'rb'))

    # append W1_Pre, W1_Post
    # fid = open(os.path.join(path_vul, 'vul_function_non_res.p'),'wb')
    # cPickle.dump(vul_function, fid)

    # bring GAR vulnerability
    # mmi_range = np.arange(4.0, 8.02, 0.02)
    # gar_vul_file = os.path.join(path_gar_vul, 'data_final.p')
    # gar_vul = cPickle.load(open(gar_vul_file,'rb'))
    bldg_mapping = pd.read_csv(os.path.join(path_vul, 'bldg_class_mapping_non_res.csv'))

    # In[28]:

    mapping_dic = {}
    for i, value in bldg_mapping.iterrows():
       mapping_dic[value['NEXIS_CONS']] = value['MAPPING2']

    site['VUL_CLASS'] = site.apply(lambda row: assign_vulnerability_class(row, mapping_dic), axis=1)

    # check all the bldgs are included in the mapping dic
    for x in site['VUL_CLASS'].unique():
       if x not in vul_function:
           print('{} is not mappable'.format(x))

    # read ground motion
    soil = np.load(os.path.join(gm_path, '{}_motion'.format(site_tag), 'soil_SA.npy'))

     # In[82]:

    SA03= soil[0, 0, 0, :, 0, 1]

    mmi_by_worden_from_SA03 = worden_et_al(SA03*980.0, 0.3, 1)

    # mmi_by_worden_from_SA03.min(), mmi_by_worden_from_SA03.max()

    # mmi_by_AK_from_SA03 = rsa2mmi_array(SA03, period=0.3)
    # mmi_by_AK_from_SA03.min(), mmi_by_AK_from_SA03.max()

    site['MMI'] = mmi_by_worden_from_SA03
   

    # vulnerability class
    # URML 1945 for pivotal year, otherwise 1996
    # C2H takes medium 
    # other types Low for Pre and Medium for Post

    """
    # plot vulnerability function
    # In[41]:
    gar_mmi = np.arange(4.0, 11.0, 0.05)

    # In[40]:
    vul_function = dict()
    for item in site['VUL_CLASS'].unique():
        temp = item.split('_')
        if temp[0] == 'C2H':
            vul_function[item] = gar_vul[temp[0]]['Medium']['ratio']        
        else:
            if temp[1] == 'Pre':
                vul_function[item] = gar_vul[temp[0]]['Low']['ratio']        
            else:
                vul_function[item] = gar_vul[temp[0]]['Medium']['ratio']        

    # manually input
    vul_function['URML_Pre'] = stats.lognorm.cdf(gar_mmi, 0.16, scale=8.0)
    vul_function['URML_Post'] = stats.lognorm.cdf(gar_mmi, 0.18, scale=8.74)

    # take URML_Pre1945 for non-residential URML_Pre1945
    plt.plot(gar_mmi, compute_vulnerability_res(gar_mmi, 'URML_Pre'))
    plt.xlim([4, 8])
    plt.grid(1)


    # In[69]:

    vul_function['URML_Pre'] = compute_vulnerability(gar_mmi, 'URML_Pre')


    # In[48]:

    plt.figure()
    for item, value in vul_function.iteritems():
        if 'Post' in item:
            plt.plot(gar_mmi, value, label=item)
        else:
            pass
    plt.xlim([4, 8])
    plt.ylim([0, 0.6])
    plt.grid(1)
    plt.legend(loc=2)
    plt.savefig('./vul_non_residential_Post_v1.png')

    plt.figure()
    for item, value in vul_function.iteritems():
        if 'Pre' in item:
            plt.plot(gar_mmi, value, label=item)
        else:
            pass
    plt.xlim([4, 8])
    plt.ylim([0, 0.6])
    plt.grid(1)
    plt.legend(loc=2)
    plt.savefig('./vul_non_residential_Pre_v1.png')
    """

    # In[80]:
    site['LOSS_RATIO'] = site.apply(lambda row: np.interp(row['MMI'], gar_mmi, vul_function[row['VUL_CLASS']]), axis=1)

    # In[82]:
    """
    # MEAN LOSS RATIO by SA1
    grouped = site.groupby('SA1_CODE')
    mean_loss_ratio_by_SA1 = grouped['LOSS_RATIO'].mean()
    mean_loss_ratio_by_SA1.fillna(0, inplace=True)
    mean_loss_ratio_by_SA1.columns = ['SA1_CODE', 'LOSS_RATIO']
    file_ = os.path.join(output_path,'mean_loss_ratio_by_SA1.csv')
    mean_loss_ratio_by_SA1.to_csv(file_)
    print('{} is created'.format(file_))
    """

    # In[86]:
    # sample loss ratio assuming gamma distribution with constant cov
    cov = 1.0
    #okay = data[~data['LOSS_RATIO'].isnull()].index
    #mean_lratio = data.loc[okay, 'LOSS_RATIO'].values
    #pop = data.loc[okay, 'POPULATION'].values

    np.random.seed(99)
    # np.array(nsample, nbldgs)
    sample = sample_vulnerability(site['LOSS_RATIO'].values, nsample=nsample,
                                  cov=cov)

    # assign damage state
    damage_labels = ['no', 'slight', 'moderate', 'extensive', 'complete']
    damage_thresholds = [-1.0, 0.02, 0.1, 0.5, 0.8, 1.1]

    # In[87]:
    df_damage = np.digitize(np.transpose(sample), damage_thresholds)

    # In[91]:
    df_damage = pd.DataFrame(df_damage)

    # In[89]:
    # bldg_types = site['VUL_CLASS'].unique()

    # In[97]:
    #bldg_dmg_count = pd.DataFrame(0.0, columns=bldg_types,
    #                              index=range(1, len(damage_labels) + 2))
    #for bldg_str, grouped in df_damage.groupby(['VUL_CLASS']):
    #    bldg_dmg_count[bldg_str] = grouped.apply(pd.value_counts, axis=0).fillna(0.0).mean(axis=1)

    bldg_dmg_count = df_damage.apply(pd.value_counts, axis=0).fillna(0.0).mean(axis=1)
    
    file_ = os.path.join(output_path, 'bldg_dmg_count.csv')
    bldg_dmg_count.to_csv(file_)
    print('{} is created'.format(file_))

    """
    # bldg_dmg_count by suburb
    sub = site['SUBURB'].unique().tolist()
    bldg_dmg_count_by_sub = pd.DataFrame(0.0, columns=sub,
                                         index=range(1, len(damage_labels) + 2))
    for sub_str, grouped in site.groupby('SUBURB'):
        bldg_dmg_count_by_sub[sub_str] = df_damage.ix[grouped.index].apply(pd.value_counts, axis=0).fillna(0.0).mean(axis=1)

    file_ = os.path.join(output_path, 'bldg_dmg_count_by_sub.csv')
    bldg_dmg_count_by_sub.to_csv(file_)
    print('{} is created'.format(file_))
    """

   # casualty
    idx = site[site['POPULATION'].notnull()].index.tolist()
    sel = df_damage.loc[idx]

    bldg_class_for_casualty = [x.split('_')[0] for x in site.loc[idx, 'VUL_CLASS'].tolist()]
    sel['VUL_CLASS'] = bldg_class_for_casualty

    casualty_rate = read_hazus_casualty_data(hazus_data_path,
                                             selected_bldg_class=bldg_class_for_casualty)

    # read hazus collapse rate data
    collapse_rate = read_hazus_collapse_rate(hazus_data_path,
                                             selected_bldg_class=bldg_class_for_casualty)

    # assign casualty rate by damage state
    # casualty{'Severity'}.DataFrame
    casualty = assign_casualty(sel, casualty_rate)

    # save casualty
    # for severity in casualty.keys():
    #     file_ = os.path.join(data_path, '%s_%s.csv' % ('casualty_rate', severity))
    #     casualty[severity].to_csv(file_, index=False)
    #     print "%s is created" % file_

    # multiply casualty with population
    casualty_number = pd.DataFrame(index=range(nsample),
                                   columns=casualty_rate.minor_axis)
    for severity in casualty_rate.minor_axis:
        value_ = 0.01*casualty[severity][range(nsample)].multiply(site.loc[idx, 'POPULATION'].values, axis=0)
        casualty_number.loc[:, severity] = value_.sum(axis=0)

    # print casualty_number.mean(axis=0)
    file_ = os.path.join(output_path, 'casualty_number.csv')
    casualty_number.to_csv(file_, index=False)
    print('{} is created'.format(file_))
 

if __name__ == '__main__':

    # environment
    import sys
    if sys.platform == 'darwin':
        pdir = '/Users/hyeuk/Projects'
    else:
        pdir = '/nas/users/u65242/unix/Projects'
        
    project_tag = 'scenario_Guildford'
    site_tag = 'perth_school'
    gm_tag = 'gm_school_Mw5.6D7'
    site_csv_file = os.path.join(pdir, project_tag, 'input',
                                 'Guildford_exposure_school.csv')
    output_path = os.path.join(pdir, project_tag, 'school')
    path_vul = os.path.join(pdir, project_tag, 'input')
    hazus_data_path = os.path.join(pdir, 'scenario_Sydney/data/hazus')

    main(pdir, project_tag, site_tag, gm_tag, site_csv_file, path_vul, hazus_data_path, output_path, nsample=10)

