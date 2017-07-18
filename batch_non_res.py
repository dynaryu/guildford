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

"""
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

import pylab
pylab.rcParams['figure.figsize'] = (8.0, 6.0)
pylab.rcParams['figure.dpi'] = 300
pylab.rcParams['font.size'] = 12
pylab.rcParams['legend.numpoints'] = 1
"""

# In[3]:

working_path = os.path.join(os.path.expanduser("~"),'Projects/eq_victoria')


# In[78]:

def compute_vulnerability_res(mmi, bldg_class):

    def inv_logit(x):
        return np.exp(x)/(1.0 + np.exp(x))

    def compute_mu(mmi, **kwargs):

        coef = {
            "t0": -8.56,
            "t1": 0.92,
            "t2": -4.82,
            "t3": 2.74,
            "t4": 0.49,
            "t5": -0.31}

        flag_timber = kwargs['flag_timber']
        flag_pre = kwargs['flag_pre']

        mu = coef["t0"] + coef["t1"]*mmi + coef["t2"]*flag_timber + coef["t3"]*flag_pre + coef["t4"]*flag_timber*mmi + coef["t5"]*flag_pre*mmi
        return mu

    flag_timber = 'Timber' in bldg_class
    flag_pre = 'Pre' in bldg_class

    # correction of vulnerability suggested by Mark
    result = np.zeros_like(mmi)
    tf = mmi < 5.5
    prob55 = inv_logit(compute_mu(5.5, flag_timber=flag_timber, flag_pre=flag_pre))
    result[tf] = np.interp(mmi[tf], [4.0, 5.5], [0.0, prob55], left=0.0)
    mu = compute_mu(mmi[~tf], flag_timber=flag_timber, flag_pre=flag_pre)
    result[~tf] = inv_logit(mu)
    return result


# In[75]:

def assign_vulnerability_class(df, mapping_dic):

    bldg_ = mapping_dic[df['NEXIS_CONSTRUCTION_TYPE']]
    year_ = convert_to_float(df['NEXIS_YEAR_BUILT'])

    if bldg_ == 'URML':
        if  year_ <= 1946: 
            tail = 'Pre'
        else:
            tail = 'Post'
    else:
        if year_ <= 1996: 
            tail = 'Pre'
        else:
            tail = 'Post'
    return pd.Series({'VUL_CLASS': '{}_{}'.format(bldg_, tail)})


# # In[4]:

# def compute_vulnerability(mmi, bldg_class):
#     gar_mmi = np.arange(4.0, 11.0, 0.05)
#     return np.interp(mmi, gar_mmi, vul_function[bldg_class])


# In[9]:

def sample_vulnerability(mean_lratio, nsample=1000, cov=1.0):

    """
    The probability density function for `gamma` is::

        gamma.pdf(x, a) = lambda**a * x**(a-1) * exp(-lambda*x) / gamma(a)

    for ``x >= 0``, ``a > 0``. Here ``gamma(a)`` refers to the gamma function.

    The scale parameter is equal to ``scale = 1.0 / lambda``.

    `gamma` has a shape parameter `a` which needs to be set explicitly. For
    instance:

        >>> from scipy.stats import gamma
        >>> rv = gamma(3., loc = 0., scale = 2.)

    shape: a
    scale: b
    mean = a*b
    var = a*b*b
    cov = 1/sqrt(a) = 1/sqrt(shape)
    shape = (1/cov)^2
    """

    shape_ = np.power(1.0/cov, 2.0)
    scale_ = mean_lratio/shape_
    sample = stats.gamma.rvs(shape_, loc=0, scale=scale_,
                             size=(nsample, len(mean_lratio)))

    sample[sample > 1] = 1.0

    return sample


# In[25]:

def convert_to_float(string):
    val = string.split('-')[-1]
    try:
        float(val)
        return float(val)
    except ValueError:
        #print "Not a Number"
        return None


def main(pdir, project_tag, site_tag, gm_tag, site_csv_file, path_vul, output_path, nsample=10):

    gm_path = os.path.join(pdir, project_tag, gm_tag)

    # Read Non-Residential Bldgs
    site = pd.read_csv(site_csv_file)
    #fields_type = [x[1] for x in sf.fields[1:]]

    # read vulnerability for non residential bldgs
    gar_mmi = np.arange(4.0, 11.0, 0.05)
    vul_function = cPickle.load(open(os.path.join(path_vul, 'vul_function_non_res.p'),'rb'))

    # In[4]:

    # bring GAR vulnerability
    # mmi_range = np.arange(4.0, 8.02, 0.02)
    # gar_vul_file = os.path.join(path_gar_vul, 'data_final.p')
    # gar_vul = cPickle.load(open(gar_vul_file,'rb'))
    bldg_mapping = pd.read_csv(os.path.join(path_vul, 'bldg_class_mapping_non_res.csv'))

    # In[28]:

    mapping_dic = {}
    for i, value in bldg_mapping.iterrows():
       mapping_dic[value['NEXIS_CONS']] = value['MAPPING2']

    # check all the bldgs are included in the mapping dic
    # for x in site['NEXIS_CONSTRUCTION_TYPE'].unique():
    #    if x not in mapping_dic:
    #        print('{} is not mappable'.format(x))

    # read ground motion
    soil = np.load(os.path.join(gm_path, '{}_motion'.format(site_tag), 'soil_SA.npy'))


     # In[82]:

    SA03= soil[0, 0, 0, :, 0, 1]

    mmi_by_worden_from_SA03 = worden_et_al(SA03*980.0, 0.3, 1)

    # mmi_by_worden_from_SA03.min(), mmi_by_worden_from_SA03.max()

    # mmi_by_AK_from_SA03 = rsa2mmi_array(SA03, period=0.3)
    # mmi_by_AK_from_SA03.min(), mmi_by_AK_from_SA03.max()

    site['MMI'] = mmi_by_worden_from_SA03
   
    site['VUL_CLASS'] = site.apply(lambda row: assign_vulnerability_class(row, mapping_dic), axis=1)

    # check all the bldgs are included in the mapping dic
    for x in site['VUL_CLASS'].unique():
       if x not in vul_function:
           print('{} is not mappable'.format(x))

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

    # MEAN LOSS RATIO by SA1
    grouped = site.groupby('SA1_CODE')
    mean_loss_ratio_by_SA1 = grouped['LOSS_RATIO'].mean()
    mean_loss_ratio_by_SA1.fillna(0, inplace=True)
    mean_loss_ratio_by_SA1.columns = ['SA1_CODE', 'LOSS_RATIO']
    file_ = os.path.join(output_path,'mean_loss_ratio_by_SA1.csv')
    mean_loss_ratio_by_SA1.to_csv(file_)
    print('{} is created'.format(file_))

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
    df_damage['VUL_CLASS'] = site['VUL_CLASS']

    # In[89]:
    bldg_types = site['VUL_CLASS'].unique()

    # In[97]:
    bldg_dmg_count = pd.DataFrame(0.0, columns=bldg_types,
                                  index=range(1, len(damage_labels) + 2))
    for bldg_str, grouped in df_damage.groupby(['VUL_CLASS']):
        bldg_dmg_count[bldg_str] = grouped.apply(pd.value_counts, axis=0).fillna(0.0).mean(axis=1)

    file_ = os.path.join(output_path, 'bldg_dmg_count.csv')
    bldg_dmg_count.to_csv(file_)
    print('{} is created'.format(file_))

    # bldg_dmg_count by suburb
    sub = site['SUBURB'].unique().tolist()
    bldg_dmg_count_by_sub = pd.DataFrame(0.0, columns=sub,
                                         index=range(1, len(damage_labels) + 2))
    for sub_str, grouped in site.groupby('SUBURB'):
        bldg_dmg_count_by_sub[sub_str] = df_damage.ix[grouped.index].apply(pd.value_counts, axis=0).fillna(0.0).mean(axis=1)

    file_ = os.path.join(output_path, 'bldg_dmg_count_by_sub.csv')
    bldg_dmg_count_by_sub.to_csv(file_)
    print('{} is created'.format(file_))


if __name__ == '__main__':

    # environment
    import sys
    if sys.platform == 'darwin':
        pdir = '/Users/hyeuk/Projects'
    else:
        pdir = '/nas/users/u65242/unix/Projects'
        
    project_tag = 'scenario_Guildford'
    site_tag = 'perth_non_res_bldg'
    gm_tag = 'gm_non_res_bldg_Mw5.6D7'
    site_csv_file = os.path.join(pdir, project_tag, 'input',
                                 'NEXISv7_WA_CommercialIndustrial_siteclass_5GPER.csv')
    output_path = os.path.join(pdir, project_tag, 'non_res')
    path_vul = os.path.join(pdir, project_tag, 'input')

    main(pdir, project_tag, site_tag, gm_tag, site_csv_file, path_vul, output_path, nsample=10)

