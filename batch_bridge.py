import os
import pandas as pd
import numpy as np
from scipy import stats

"""
import matplotlib
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import pylab

pylab.rcParams['figure.figsize'] = (10.0, 8.0)
pylab.rcParams['legend.numpoints'] = 1
"""

def get_k3d(A, B, no_span):
    try:
        K3d = 1 + A/(no_span-B)
    except ZeroDivisionError:
        K3d = 1 + A
    return K3d


def get_kshape(Ishape, sa03, sa10):
    if Ishape:
        Kshape = 2.5*sa10/sa03
    else:
        Kshape  = 1.0
    return Kshape


def cal_pe(b_class, sa03, sa10, no_span, skew):
    
    eq_k3d, Ishape = bridge_param.loc[b_class, 'K3D'], bridge_param.loc[b_class, 'Ishape']
    A, B = bridge_k3d_coeffs.loc[eq_k3d, 'A'],  bridge_k3d_coeffs.loc[eq_k3d, 'B']
    k3d = get_k3d(A, B, no_span)
    kshape = get_kshape(Ishape, sa03, sa10)
    Kskew = np.sqrt(np.sin(np.deg2rad(90.0-skew)))
    ds = ['Slight','Moderate','Extensive','Complete']
    
    pe_by_ds = []
    med = kshape*bridge_param.loc[b_class, 'Slight']
    pe = stats.lognorm.cdf(sa10, 0.6, scale=med)
    pe_by_ds.append(pe)

    for x in ds[1:]:
        med = Kskew*k3d*bridge_param.loc[b_class, x]
        pe = stats.lognorm.cdf(sa10, 0.6, scale=med)
        pe_by_ds.append(pe)

    return pe_by_ds


# In[129]:

def cal_pe_by_row(row):
    
    b_class = row['STRUCTURE_CLASSIFICATION']
    sa03 = row['SA03']
    sa10 = row['SA10']
    no_span = row['SPAN']
    skew = row['SKEW']
    
    eq_k3d, Ishape = bridge_param.loc[b_class, 'K3D'], bridge_param.loc[b_class, 'Ishape']
    A, B = bridge_k3d_coeffs.loc[eq_k3d, 'A'],  bridge_k3d_coeffs.loc[eq_k3d, 'B']
    k3d = get_k3d(A, B, no_span)
    kshape = get_kshape(Ishape, sa03, sa10)
    Kskew = np.sqrt(np.sin(np.deg2rad(90.0-skew)))
    ds = ['Slight','Moderate','Extensive','Complete']
    
    pe_by_ds = []
    med = kshape*bridge_param.loc[b_class, 'Slight']
    pe = stats.lognorm.cdf(sa10, 0.6, scale=med)
    pe_by_ds.append(pe)

    for x in ds[1:]:
        med = Kskew*k3d*bridge_param.loc[b_class, x]
        pe = stats.lognorm.cdf(sa10, 0.6, scale=med)
        pe_by_ds.append(pe)

    return np.array(pe_by_ds)


def main(pdir, project_tag, site_tag, gm_tag, nsample=1000):

    global bridge_param, bridge_k3d_coeffs
    
    gm_path = os.path.join(pdir, project_tag, gm_tag)
    output_path = os.path.join(pdir, project_tag, gm_tag)

    # read sitedb data 
    site = pd.read_csv(os.path.join(pdir, project_tag, 'input/bridgedb_{}.csv'.format(site_tag)))

    # ground motion
    gm = np.load(os.path.join(gm_path, '{}_motion'.format(site_tag), 'soil_SA.npy'))

    # append SA03, SA10
    site['SA03'] = gm[0, 0, 0, :, 0, 1]
    site['SA10'] = gm[0, 0, 0, :, 0, 2]


    # HAZUS Methodology
    try:
        bridge_param = pd.read_csv(
            os.path.join(pdir, project_tag, 'input', 'bridge_classification_damage_params.csv'), index_col=0)
    except IOError:
        bridge_param = pd.read_csv(
            os.path.join(eqrm_data_path, 'bridge_classification_damage_params.csv'), index_col=0)

    # In[34]:
    bridge_k3d_coeffs = pd.read_csv(os.path.join(eqrm_data_path, 'bridge_k3d_coefficients.csv'), index_col=0)

    """
    # In[36]:

    bridge_k3d_coeffs.loc['EQ1']


    # In[31]:

    bridge_param.tail()


    # In[37]:
    """



    # In[132]:

    #cal_pe_by_row(site.loc[0]).shape


    # compute pe_ds
    damage = []
    for i, row in site.iterrows():
        damage.append(cal_pe_by_row(row))
    damage = np.array(damage)


    """
    # ## HAZUS HWB17 validation

    # In[115]:

    cal_pe('HWB17', 2.1, 0.43, no_span=3.0, skew=32.0)

    # ## validation against EQRM output

    # In[116]:

    cal_pe_by_row(site.loc[0])


    # In[117]:

    eqrm_output = pd.read_csv('../bridge_Mw5.6D7/perth_structural_damage.txt')


    # In[118]:

    eqrm_output.head()


    # In[120]:

    site.loc[0, 'PE']
    """

    # estimate no. of damage 
    # nsample = 100
    sampled = np.random.uniform(size=(damage.shape[0], nsample))


    # In[143]:
    df_ds = pd.DataFrame(0, index=range(damage.shape[0]), columns=range(nsample))
    for irow in range(damage.shape[0]):
        df_ds.loc[irow, :] = np.digitize(sampled[irow, :], damage[irow, :])


    # In[162]:
    df_ds.apply(pd.value_counts, axis=0).fillna(0.0).mean(axis=1).to_csv(
        os.path.join(output_path, 'no_damage.csv'))

    # In[171]:
    for i, item in enumerate(['slight', 'moderate', 'extensive', 'complete']):
        site['pe_{}'.format(item)] = damage[:, i]

    # save to csv file
    site.to_csv(os.path.join(output_path, 'result_pe.csv'), index=False)


# In[ ]:

if __name__ == '__main__':

    # environment
    pdir = '/Users/hyeuk/Projects'
    project_tag = 'scenario_Guildford'
    site_tag = 'perth'
    gm_tag = 'bridge_Mw5.6D7'

    eqrm_data_path = os.path.join(pdir, 'eqrm/resources/data')

    main(pdir, project_tag, site_tag, gm_tag, nsample=1000)



