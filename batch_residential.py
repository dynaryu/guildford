import os
import pandas as pd
import numpy as np
from collections import OrderedDict
from scipy import stats

from eqrm_code.worden_et_al import worden_et_al
from eqrm_code.RSA2MMI import rsa2mmi_array


"""
import matplotlib
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import pylab

pylab.rcParams['figure.figsize'] = (10.0, 8.0)
pylab.rcParams['legend.numpoints'] = 1
"""

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


def read_hazus_collapse_rate(hazus_data_path, selected_bldg_class=None):
    """
    read hazus collapse rate parameter values
    """

    # read collapse rate (table 13.8)
    fname = os.path.join(hazus_data_path, 'hazus_collapse_rate.csv')
    collapse_rate = pd.read_csv(fname, skiprows=1, names=['Bldg type', 'rate'],
                                index_col=0, usecols=[1, 2])
    collapse_rate = collapse_rate.to_dict()['rate']

    if selected_bldg_class is not None:
        removed_bldg_class = (set(collapse_rate.keys())).difference(set(
            selected_bldg_class))
        [collapse_rate.pop(item) for item in removed_bldg_class]

    return collapse_rate


def read_hazus_casualty_data(hazus_data_path, selected_bldg_class=None):
    """
    read hazus casualty parameter values
    """

    # read indoor casualty (table13.3 through 13.7)
    severity_list = ['Severity{}'.format(i) for i in range(1, 5)]
    list_ds = ['slight', 'moderate', 'extensive', 'complete', 'collapse']
    colname = ['Bldg type'] + severity_list

    dic_ = dict()
    for ds in list_ds:

        file_ = 'hazus_indoor_casualty_{}.csv'.format(ds)
        fname = os.path.join(hazus_data_path, file_)

        # tmp = pd.read_csv(fname, skiprows=1, header=None)
        tmp = pd.read_csv(fname, skiprows=1,
                          names=colname, usecols=[1, 2, 3, 4, 5], index_col=0)
        if selected_bldg_class is not None:
            okay = tmp.index.isin(selected_bldg_class)
            dic_[ds] = tmp.ix[okay]
        else:
            dic_[ds] = tmp

    casualty_rate = pd.Panel(dic_, items=list_ds)
    casualty_rate['none'] = 0.0
    list_ds.insert(0, 'none')
    return casualty_rate[list_ds] # re-order the item


def assign_damage_state(data, sample, collapse_rate, damage_thresholds):
    """
    assign damage state given damage state thresholds

    """

    df_damage = np.digitize(np.transpose(sample), damage_thresholds)
    df_damage = pd.DataFrame(df_damage)
    df_damage['BLDG_CLASS'] = data['BLDG_CLASS']

    # df_damage = pd.DataFrame(index=data.index, columns=range(nsample))

    # # assign damage by loss ratio
    # for i in range(nsample):
    #     df_damage[i] = pd.cut(sample[i, :], damage_thresholds,
    #                           labels=damage_labels)
    #     df_damage[i] = df_damage[i].cat.add_categories(['collapse'])


    # assign collapse
    for name, group in df_damage.groupby('BLDG_CLASS'):

        prob_collapse = collapse_rate[name]*1.0e-2

        idx_group = group.index
        group_array = group.values[:, :-1]

        (idx_complete, idy_complete) = np.where(group_array == 5) # complte
        ncomplete = len(idx_complete)
        # temp = np.random.choice(['complete', 'collapse'], size=ncomplete,
        #                         p=[1-prob_collapse, prob_collapse])
        temp = np.random.choice([5, 6], size=ncomplete,
                                p=[1-prob_collapse, prob_collapse])

        idx_collapse = np.where(temp == 6)[0]
        for i in idx_collapse:
            df_damage.loc[idx_group[idx_complete[i]], idy_complete[i]] = 6

    return df_damage

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


def plot_vulnerabilty():
    """ plot vulnerability """
    mmi_range = np.arange(4.0, 10.0, 0.05)

    vul = OrderedDict()
    for bldg in ['Timber_Pre1945', 'Timber_Post1945', 'URM_Pre1945',
                 'URM_Post1945']:
        tmp = []
        for val in mmi_range:
            temp = {'mmi': val, 'BLDG_CLASS': bldg}
            tmp.append(compute_vulnerability(temp))
        vul[bldg] = np.array(tmp)

    line_type = ['b--', 'b-', 'r--', 'r-']
    plt.figure()
    for bldg, line_ in zip(vul.keys(), line_type):
        label_str = bldg.replace('_', ':')
        plt.plot(mmi_range, vul[bldg],  line_, label=label_str)

    plt.legend(loc=2)
    plt.grid(1)
    plt.xlabel('MMI')
    plt.ylabel('Loss ratio')
    plt.xlim([4, 8.0])
    plt.ylim([0, 0.3])

def compute_vulnerability(row):

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

        mu = coef["t0"] + \
             coef["t1"]*mmi + \
             coef["t2"]*flag_timber + \
             coef["t3"]*flag_pre + \
             coef["t4"]*flag_timber*mmi + \
             coef["t5"]*flag_pre*mmi
        return mu

    flag_timber = 'Timber' in row['BLDG_CLASS']
    flag_pre = 'Pre' in row['BLDG_CLASS']

    # correction of vulnerability suggested by Mark
    if row['mmi'] < 5.5:
        prob55 = inv_logit(compute_mu(5.5,
                           flag_timber=flag_timber,
                           flag_pre=flag_pre))
        return np.interp(row['mmi'], [4.0, 5.5], [0.0, prob55], left=0.0)
    else:
        mu = compute_mu(row['mmi'],
                        flag_timber=flag_timber,
                        flag_pre=flag_pre)
        return inv_logit(mu)

def IPE_by_Trevor(rhyp, mw):

    from math import log10, sqrt, erf

    c0 = 1.03561713242
    c1 = 3.52596119278
    c2 = -2.54489244045
    c3 = 0.000941207727714
    rref = 5.0
    xh = 50.0
    h1 = 0.191733112559
    h2 = 0.148565183719
    h3 = 0.0316028130321
    
#     mmi = np.zeros_like(rhyp)
#     mmi[rhyp <= xh] = c0 * mw + c1 + c2 * log10(sqrt(rhyp[rhyp <= xh]**2 + rref**2)) + h1*erf((rhyp[rhyp <= xh]-7.0)/(h2*sqrt(2))) + h3
#     mmi[rhyp > xh] = c0 * mw + c1 + c2 * log10(sqrt(rhyp[rhyp > xh]**2 + rref**2)) + c3 * (rhyp[rhyp > xh] - xh) + h1*erf((rhyp[rhyp > xh]-7.0)/(h2*sqrt(2))) + h3
    mmi = np.zeros_like(rhyp)
    idx = rhyp > xh
    if sum(idx):
        mmi[idx] = c0 * mw + c1 + c2 * log10(sqrt(rhyp[idx]**2 + rref**2)) + c3 * (rhyp[idx] - xh) + h1*erf((rhyp[idx]-7.0)/(h2*sqrt(2))) + h3
    
    idx = rhyp <= xh
    if sum(idx):
        mmi[idx] = c0 * mw + c1 + c2 * log10(sqrt(rhyp[idx]**2 + rref**2)) + h1*erf((rhyp[idx]-7.0)/(h2*sqrt(2))) + h3

    return mmi


def convert_to_float(string):
    val = string.split('-')[-1]
    try:
        float(val)
        return float(val)
    except ValueError:
        # print "Not a Number"
        return None

def assign_res_class(row):
    URM_list = ['URMLTILE', 'URMLMETAL', 'URMMTILE', 'URMMMETAL']
    tf_URM = row['GA_STRUCTU'] in URM_list

    if tf_URM:
        bldg = 'URM'
    else:
        bldg = 'Timber'

    idx_1945 = convert_to_float(row['YEAR_BUILT'])
        
    if idx_1945 <=1946 or None:
        age = 'Pre1945'
    else:
        age = 'Post1945'
        
    return '{}_{}'.format(bldg, age)

"""
# In[35]:

convert_to_float('1788 - 1939')


# In[36]:

convert_to_float('Unknown') is None
"""

def main(gm_path, site_tag, site_csv_file, hazus_data_path, output_path, nsample=1000):

    path_gm = os.path.join(gm_path, '{}_motion'.format(site_tag))

    # read sitedb data 
    site = pd.read_csv(site_csv_file)

    """
    # In[18]:

    site.head()


    # In[19]:

    site.shape


    # In[20]:

    site.columns
    """

    """
    # In[70]:

    assign_res_class(site.loc[1])
    """

    # BLDG_CLASS added
    site['BLDG_CLASS'] = site.apply(assign_res_class, axis=1)


    """
    # In[74]:
    site['BLDG_CLASS'].value_counts()


    # In[129]:

    site['POPULATION'].sum()
    """

    # ground motion
    soil = np.load(os.path.join(path_gm, 'soil_SA.npy'))
    periods = np.load(os.path.join(path_gm, 'atten_periods.npy'))

    """
    # In[80]:

    soil.shape


    # In[81]:

    periods
    """

    # In[82]:

    SA03= soil[0, 0, 0, :, 0, 1]

    """
    # In[84]:

    SA03.min(), SA03.max()
    """

    mmi_by_worden_from_SA03 = worden_et_al(SA03*980.0, 0.3, 1)

    # mmi_by_worden_from_SA03.min(), mmi_by_worden_from_SA03.max()

    # mmi_by_AK_from_SA03 = rsa2mmi_array(SA03, period=0.3)
    # mmi_by_AK_from_SA03.min(), mmi_by_AK_from_SA03.max()

    site['mmi'] = mmi_by_worden_from_SA03

    """
    # In[92]:

    dist = np.loadtxt(os.path.join('../gm_res_bldg_Mw5.6D7/', 'perth_res_bldg_distance_rup.txt'), skiprows=1)


    # In[93]:

    dist.shape


    # In[94]:
       
    # In[95]:

    dist.min(), dist.max()


    # In[96]:

    # mmi_by_IPE = IPE_by_Trevor(np.array([4.0, 100.0]), 5.6)


    # In[97]:

    mmi_by_IPE
    """


    # vulnerability

    # In[124]:

    # plot_vulnerabilty()


    # In[100]:

    # compute_vulnerability(site.loc[30])


    # In[112]:
    # site['mmi'].groupby(pd.cut(site['mmi'], np.arange(3.0, 8.5, 0.5))).count()


    # In[125]:
    site['loss_ratio'] = site.apply(compute_vulnerability, axis=1)


    # In[127]:
    # site['loss_ratio'].groupby(pd.cut(site['loss_ratio'], np.arange(0, 0.4, 0.05))).count()

    # ## casualty

    # In[ ]:

    # sample loss ratio assuming gamma distribution with constant cov
    # nsample = 10
    #nsample = 10
    cov = 1.0
    #okay = data[~data['LOSS_RATIO'].isnull()].index
    #mean_lratio = data.loc[okay, 'LOSS_RATIO'].values
    #pop = data.loc[okay, 'POPULATION'].values

    np.random.seed(99)
    # np.array(nsample, nbldgs)
    sampled = sample_vulnerability(site['loss_ratio'].values, nsample=nsample, cov=cov)


    # In[154]:

    # nsample, no_bldg
    np.save(os.path.join(output_path, 'sampled_loss_ratio.npy'), sampled)

    """
    # no. of people to be replaced loss ratio > 0.25
    sub = site['SUBURB'].unique().tolist()
    no_replaced_by_sub = pd.DataFrame(0.0, columns=sub, index=range(nsample))
    for sub_str, grouped in site.groupby('SUBURB'):
        #print sub_str, grouped.shape
        no_replaced_by_sub[sub_str] = np.dot(sampled[:, grouped.index] > 0.25,
                                             grouped['POPULATION'].values)

    no_replaced_by_sub.sum(axis=1).mean()
    """

    # no. of people replaced (loss ratio > 0.25)
    no_replaced_total = np.dot(sampled > 0.25, site['POPULATION'].values).mean()
    print("no_replaced_total is {:.0f}".format(no_replaced_total))

    # assign damage state
    damage_labels = ['no', 'slight', 'moderate', 'extensive', 'complete']
    damage_thresholds = [-1.0, 0.02, 0.1, 0.5, 0.8, 1.1]
    bldg_types = site['BLDG_CLASS'].unique()

    # fatality estimate
    # read hazus indoor casualty data
    casualty_rate = read_hazus_casualty_data(hazus_data_path,
                                             selected_bldg_class=bldg_types)

    # read hazus collapse rate data
    collapse_rate = read_hazus_collapse_rate(hazus_data_path,
                                             selected_bldg_class=bldg_types)

    df_damage = assign_damage_state(site, sampled, collapse_rate, damage_thresholds)

    bldg_dmg_count = pd.DataFrame(0.0, columns=bldg_types,
                                  index=range(1, len(damage_labels) + 2))
    for bldg_str, grouped in df_damage.groupby(['BLDG_CLASS']):
        #print bldg_str, grouped.shape
        bldg_dmg_count[bldg_str] = grouped.apply(pd.value_counts, axis=0).fillna(0.0).mean(axis=1)

    file_ = os.path.join(output_path, 'bldg_dmg_count.csv')
    bldg_dmg_count.to_csv(file_)
    print("{} is created".format(file_))

    # bldg_dmg_count by suburb
    sub = site['SUBURB'].unique().tolist()
    bldg_dmg_count_by_sub = pd.DataFrame(0.0, columns=sub,
                                         index=range(1, len(damage_labels) + 2))
    for sub_str, grouped in site.groupby('SUBURB'):
        #print sub_str, grouped.shape
        bldg_dmg_count_by_sub[sub_str] = df_damage.ix[grouped.index].apply(pd.value_counts, axis=0).fillna(0.0).mean(axis=1)

    file_ = os.path.join(output_path, 'bldg_dmg_count_by_sub.csv')
    bldg_dmg_count_by_sub.to_csv(file_)
    print("%s is created" %file_)

    # assign casualty rate by damage state
    # casualty{'Severity'}.DataFrame
    casualty = assign_casualty(df_damage, casualty_rate)

    # save casualty
    # for severity in casualty.keys():
    #     file_ = os.path.join(data_path, '%s_%s.csv' % ('casualty_rate', severity))
    #     casualty[severity].to_csv(file_, index=False)
    #     print "%s is created" % file_

    # multiply casualty with population
    casualty_number = pd.DataFrame(index=range(nsample),
                                   columns=casualty_rate.minor_axis)
    casualty_by_sub = pd.DataFrame(index=casualty_rate.minor_axis, 
                                   columns=sub)
    for severity in casualty_rate.minor_axis:
        value_ = 0.01*casualty[severity][range(nsample)].multiply(site['POPULATION'], axis=0)
        casualty_number.loc[:, severity] = value_.sum(axis=0)

        for sub_str, grouped in site.groupby('SUBURB'):
            #print sub_str, grouped.shape
            casualty_by_sub.loc[severity, sub_str] = value_.ix[grouped.index].sum(axis=0).mean()
    # print casualty_number.mean(axis=0)

    file_ = os.path.join(output_path, 'casualty_by_sub.csv')
    casualty_by_sub.to_csv(file_, index=False)
    print('{} is created'.format(file_))

    file_ = os.path.join(output_path, 'casualty_number.csv')
    casualty_number.to_csv(file_, index=False)
    print('{} is created'.format(file_))

if __name__ == '__main__':

    # environment
    pdir = '/Users/hyeuk/Projects'
    gm_path = os.path.join(pdir, 'scenario_Guildford', 'gm_res_bldg_Mw5.6D7')
    site_tag = 'perth_res_bldg'
    site_csv_file = os.path.join(pdir, 'scenario_Guildford/input',
                                 'Perth_Residential_Earthquake_Exposure_201607_EQRM.csv')
    hazus_data_path = os.path.join(pdir, 'scenario_Sydney/data/hazus')
    output_path = os.path.join(pdir, 'scenario_Guildford', 'residential')

    main(gm_path, site_tag, site_csv_file, hazus_data_path, output_path, nsample=10)
