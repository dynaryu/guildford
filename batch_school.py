
# coding: utf-8

# In[1]:

import pandas as pd
import os
import shapefile
#from shapely.geometry import Polygon
#from shapely.geometry import Point
import numpy as np
from scipy import stats
from eqrm_code.RSA2MMI import rsa2mmi_array


# In[2]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

import pylab
pylab.rcParams['figure.figsize'] = (8.0, 6.0)
pylab.rcParams['figure.dpi'] = 300
pylab.rcParams['font.size'] = 12
pylab.rcParams['legend.numpoints'] = 1


# In[3]:

working_path = os.path.join(os.path.expanduser("~"),'Projects/eq_victoria')
data_path = os.path.join(working_path, 'exposure')


# In[12]:

get_ipython().magic(u'pinfo pd.read_excel')


# In[13]:

school = pd.read_excel(os.path.join(data_path, 'Copy of schools - earthquake - data request 20160902 GA modified.xlsx'), 
                       sheetname='GA assessment', skiprows=[0, 1])


# In[14]:

school.columns


# In[16]:

school.shape


# In[19]:

(school['HAZUS class'].notnull()).sum()


# In[20]:

school['HAZUS class'].value_counts()


# In[21]:

school['GA code'].value_counts()


# In[22]:

school['Era'].value_counts()


# In[23]:

(school['Value'].notnull()).sum()


# In[24]:

school[['Lat', 'Long']].to_csv(os.path.join(data_path, 'school_point.csv'), index=False)


# In[25]:

gm_total = np.load('../gm_school_Mw5.2D10/mel_school_motion/soil_SA.npy')
gm_total.shape


# In[26]:

gm_total = gm_total[0, 0, 0, :, 0, :]
mmi = rsa2mmi_array(gm_total[:, 2], period=1.0)


# In[27]:

mmi.min(), mmi.max()


# In[37]:

sel.shape


# In[28]:

school['mmi'] = mmi


# In[29]:

school['HAZUS class'].unique()


# In[31]:

school['GA code'].unique()


# In[32]:

school['Era'].unique()


# In[33]:

bldg_mapping = pd.read_csv('../exposure/bldg_class_mapping_non_res.csv')

mapping_dic = {}
for i, value in bldg_mapping.iterrows():
    mapping_dic[value['NEXIS_CONS']] = value['MAPPING2']


# In[34]:

mapping_dic


# In[35]:

# bring GAR vulnerability

# mmi_range = np.arange(4.0, 8.02, 0.02)
import cPickle
vul_function = cPickle.load(open('/Users/hyeuk/Projects/eq_victoria/ext_vul/vul_function_non_res.p', 'rb'))


# In[36]:

gar_mmi = np.arange(4.0, 11.0, 0.05)


# In[37]:

mapping_dic_era = {'Pre-code': 'Pre',
                   'Mid-code': 'Post'}


# In[38]:

def assign_vulnerability_class_em(df):

    bldg_ = mapping_dic[df['GA code']]
    year_ = mapping_dic_era[df['Era']]

    return pd.Series({'VUL_CLASS': '{}_{}'.format(bldg_, year_)})


# In[39]:

school['VUL_CLASS'] = school.apply(assign_vulnerability_class_em, axis=1)


# In[40]:

school['VUL_CLASS'].unique()


# In[41]:

def compute_vulnerability(mmi, bldg_class):
   
    return np.interp(mmi, gar_mmi, vul_function[bldg_class])


# In[42]:

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


# In[43]:

school['LOSS_RATIO'] = school.apply(lambda row: compute_vulnerability(
                row['mmi'], row['VUL_CLASS']), axis=1)


# In[44]:

school['LOSS_RATIO'].min(), school['LOSS_RATIO'].max()


# In[45]:

# sample loss ratio assuming gamma distribution with constant cov
nsample = 1000
cov = 1.0
#okay = data[~data['LOSS_RATIO'].isnull()].index
#mean_lratio = data.loc[okay, 'LOSS_RATIO'].values
#pop = data.loc[okay, 'POPULATION'].values

np.random.seed(99)
# np.array(nsample, nbldgs)
sample = sample_vulnerability(school['LOSS_RATIO'].values, nsample=nsample,
                              cov=cov)

# assign damage state
damage_labels = ['no', 'slight', 'moderate', 'extensive', 'complete']
damage_thresholds = [-1.0, 0.02, 0.1, 0.5, 0.8, 1.1]


# In[48]:

df_ds = np.digitize(np.transpose(sample), damage_thresholds)
df_ds = pd.DataFrame(df_ds)
df_ds['VUL_CLASS'] = school['VUL_CLASS'].values
bldg_types = school['VUL_CLASS'].unique()
bldg_dmg_count = pd.DataFrame(0.0, columns=bldg_types,
                              index=range(1, len(damage_labels) + 1))
for bldg_str, grouped in df_ds.groupby(['VUL_CLASS']):
    print bldg_str, grouped.shape
    bldg_dmg_count[bldg_str] = grouped.apply(pd.value_counts, axis=0).fillna(0.0).mean(axis=1)



# In[49]:

bldg_dmg_count


# In[50]:

bldg_dmg_count.sum(axis=1)


# In[52]:

school.to_csv('../gm_school_Mw5.2D10/mean_loss_ratio_hospital.csv', index=False)


# In[53]:

bldg_dmg_count.to_csv('../gm_school_Mw5.2D10/bldg_dmg_count_hospital.csv', index=True)


# In[ ]:



