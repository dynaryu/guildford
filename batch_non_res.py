
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


# In[4]:

# bring GAR vulnerability

# mmi_range = np.arange(4.0, 8.02, 0.02)
import cPickle
gar_vul = cPickle.load(open('/Users/hyeuk/Projects/eq_victoria/ext_vul/GAR_data/data_final.p','rb'))


# In[5]:

gar_vul.keys() 


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

def assign_vulnerability_class(df):

    bldg_ = mapping_dic[df['NEXIS_CONS']]
    year_ = convert_to_float(df['NEXIS_YEAR'])

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


# In[4]:

def compute_vulnerability(mmi, bldg_class):
   
    return np.interp(mmi, gar_mmi, vul_function[bldg_class])


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


# In[5]:

# Read Non-Residential Bldgs
file_shape = '../exposure/CommIndustrial_Nexis_V7.shp'

sf = shapefile.Reader(file_shape)
shapes = sf.shapes()
records = sf.records()
fields = [x[0] for x in sf.fields[1:]]
#fields_type = [x[1] for x in sf.fields[1:]]



# In[6]:

data_frame = pd.DataFrame(records, columns=fields)


# In[7]:

data_frame['shapes'] = shapes


# In[8]:

data_frame['Longitude'] = data_frame['shapes'].apply(lambda x: x.points[0][0])


# In[9]:

data_frame['Latitude'] = data_frame['shapes'].apply(lambda x: x.points[0][1])


# In[10]:

data_frame = data_frame.drop('shapes', 1)


# In[11]:

data_frame.columns


# In[12]:

data_frame.shape


# In[103]:

data_frame['TARGET_FID'].head()


# In[16]:

data_frame.NEXIS_NO_O


# In[13]:

data_frame['SITECLASS'].unique()


# In[14]:

tf = data_frame['SITECLASS'].apply(lambda x: x in ['BC','B','CD','D','DE','C'])


# In[ ]:

data_frame.loc[tf, ['Latitude','Longitude','SITECLASS']].to_csv('./mel_non_res_par_site.csv', index=False)


# In[15]:

sel = data_frame.loc[tf, ['SUBURB', 'SA1_CODE', 'NEXIS_YEAR', 'NEXIS_CONS','NEXIS_STRU']].copy()


# In[17]:

gm = np.load('../gm_non_res_bldg_Mw5.2D10/mel_non_res_bldg_motion/soil_SA.npy')


# In[18]:

gm = gm[0, 0, 0, :, 0, :]


# In[19]:

mmi = rsa2mmi_array(gm[:, 2], period=1.0)


# In[20]:

mmi.min(), mmi.max()


# In[21]:

sel['mmi'] = mmi


# In[23]:

sel['NEXIS_YEAR'].unique()


# In[24]:

sel['NEXIS_CONS'].unique()


# In[26]:

None > 1


# In[27]:

bldg_mapping = pd.read_csv('../exposure/bldg_class_mapping_non_res.csv')


# In[28]:

mapping_dic = {}
for i, value in bldg_mapping.iterrows():
    mapping_dic[value['NEXIS_CONS']] = value['MAPPING2']


# In[29]:

mapping_dic


# In[34]:

sel.columns


# In[38]:

sel['VUL_CLASS'] = sel.apply(assign_vulnerability_class, axis=1)


# In[39]:

sel['VUL_CLASS'].unique()


# In[ ]:

# vulnerability class
# URML 1945 for pivotal year, otherwise 1996
# C2H takes medium 
# other types Low for Pre and Medium for Post


# In[40]:

vul_function = dict()
for item in sel['VUL_CLASS'].unique():
    temp = item.split('_')
    if temp[0] == 'C2H':
        vul_function[item] = gar_vul[temp[0]]['Medium']['ratio']        
    else:
        if temp[1] == 'Pre':
            vul_function[item] = gar_vul[temp[0]]['Low']['ratio']        
        else:
            vul_function[item] = gar_vul[temp[0]]['Medium']['ratio']        


# In[41]:

gar_mmi = np.arange(4.0, 11.0, 0.05)


# In[42]:

# manually input


# In[43]:

vul_function['URML_Pre'] = stats.lognorm.cdf(gar_mmi, 0.16, scale=8.0)
vul_function['URML_Post'] = stats.lognorm.cdf(gar_mmi, 0.18, scale=8.74)


# In[79]:

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


# In[70]:

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


# In[71]:

fid = open('/Users/hyeuk/Projects/eq_victoria/ext_vul/vul_function_non_res.p','wb')


# In[72]:

cPickle.dump(vul_function, fid)
fid.close()


# In[73]:

sel.columns


# In[80]:

sel['LOSS_RATIO'] = sel.apply(lambda row: compute_vulnerability(
                row['mmi'], row['VUL_CLASS']), axis=1)


# In[81]:

sel['LOSS_RATIO'].min(), sel['LOSS_RATIO'].max()


# In[82]:

# MEAN LOSS RATIO by SA1
grouped = sel.groupby('SA1_CODE')
mean_loss_ratio_by_SA1 = grouped['LOSS_RATIO'].mean()



# In[83]:

mean_loss_ratio_by_SA1.columns = ['SA1_CODE', 'LOSS_RATIO']
file_ = os.path.join(working_path,'gm_non_res_bldg_Mw5.2D10', 'mean_loss_ratio_by_SA1.csv')
mean_loss_ratio_by_SA1.to_csv(file_)
print "%s is created" %file_



# In[86]:

# sample loss ratio assuming gamma distribution with constant cov
nsample = 1000
cov = 1.0
#okay = data[~data['LOSS_RATIO'].isnull()].index
#mean_lratio = data.loc[okay, 'LOSS_RATIO'].values
#pop = data.loc[okay, 'POPULATION'].values

np.random.seed(99)
# np.array(nsample, nbldgs)
sample = sample_vulnerability(sel['LOSS_RATIO'].values, nsample=nsample,
                              cov=cov)

# assign damage state
damage_labels = ['no', 'slight', 'moderate', 'extensive', 'complete']
damage_thresholds = [-1.0, 0.02, 0.1, 0.5, 0.8, 1.1]


# In[87]:

df_ds = np.digitize(np.transpose(sample), damage_thresholds)


# In[91]:

df_ds = pd.DataFrame(df_ds)
df_ds['VUL_CLASS'] = sel['VUL_CLASS']


# In[89]:

bldg_types = sel['VUL_CLASS'].unique()


# In[97]:

bldg_dmg_count = pd.DataFrame(0.0, columns=bldg_types,
                              index=range(1, len(damage_labels) + 2))
for bldg_str, grouped in df_ds.groupby(['VUL_CLASS']):
    print bldg_str, grouped.shape
    bldg_dmg_count[bldg_str] = grouped.apply(pd.value_counts, axis=0).fillna(0.0).mean(axis=1).round()


# In[98]:

file_ = os.path.join(working_path, './gm_non_res_bldg_Mw5.2D10/bldg_dmg_count.csv')
bldg_dmg_count.to_csv(file_)
print("%s is created" %file_)


# In[96]:

bldg_dmg_count.sum(axis=0)


# In[105]:

sel['Latitude'] = data_frame.loc[tf, 'Latitude']
sel['Longitude'] = data_frame.loc[tf, 'Longitude']
sel['FID'] = data_frame.loc[tf, 'OBJECTID']


# In[106]:

# clean up data and save
file_ = os.path.join(working_path, './gm_non_res_bldg_Mw5.2D10/loss_ratio_by_bldg.csv')
sel.to_csv(file_, index=False)
print("%s is created" %file_)



# In[143]:

a = pd.read_csv('../gm_non_res_bldg_Mw5.2D10/loss_ratio_by_bldg.csv')


# In[144]:

a.head()


# In[145]:

summary_loss = pd.DataFrame(index=a['SUBURB'].unique(), columns=['LOSS_RATIO'])


# In[146]:

for sub_str, grouped in a.groupby('SUBURB'):
    summary_loss.loc[sub_str, 'LOSS_RATIO'] = (grouped['LOSS_RATIO']*grouped['NEXIS_STRU']).sum()/grouped['NEXIS_STRU'].sum()


# In[147]:

summary_loss.head()


# In[10]:

# sample loss ratio assuming gamma distribution with constant cov
nsample = 1000
cov = 1.0
#okay = data[~data['LOSS_RATIO'].isnull()].index
#mean_lratio = data.loc[okay, 'LOSS_RATIO'].values
#pop = data.loc[okay, 'POPULATION'].values

np.random.seed(99)
# np.array(nsample, nbldgs)
sample = sample_vulnerability(a['LOSS_RATIO'].values, nsample=nsample,
                              cov=cov)

# assign damage state
damage_labels = ['no', 'slight', 'moderate', 'extensive', 'complete']
damage_thresholds = [-1.0, 0.02, 0.1, 0.5, 0.8, 1.1]


# In[12]:

df_ds = np.digitize(np.transpose(sample), damage_thresholds)
df_ds = pd.DataFrame(df_ds)
df_ds['SUBURB'] = a['SUBURB']


# In[13]:

df_ds.head()


# In[148]:

summary_loss1 = pd.DataFrame(columns=a['SUBURB'].unique(), index = range(1, 6))
for sub_str, grouped in df_ds.groupby('SUBURB'):
    summary_loss1[sub_str] = grouped.apply(pd.value_counts, axis=0).fillna(0.0).mean(axis=1) 


# In[149]:

num_by_sub = a.groupby('SUBURB').apply(len)


# In[150]:

summary_loss1.head()


# In[151]:

summary_loss1t = summary_loss1.transpose()


# In[152]:

summary_loss1t['NO_BLDGS'] = num_by_sub1


# In[153]:

summary_loss1t['LOSS_RATIO'] = summary_loss['LOSS_RATIO']


# In[161]:

summary_loss1t.head()


# In[162]:

summary_loss1t.fillna(0, inplace=True)


# In[156]:

sub_list = sorted(summary_loss1t.index.tolist())


# In[163]:

summary_loss1tr = summary_loss1t.reindex(sub_list)


# In[164]:

summary_loss1tr.head()


# In[165]:

summary_loss1tr.to_csv('../gm_non_res_bldg_Mw5.2D10/summary_non_res.csv',index=True)


# In[166]:

summary_loss1tr.round().apply(sum, axis=0)


# In[176]:

get_ipython().magic(u'pinfo pd.Series.argsort')


# In[179]:

summary_loss1tr.shape


# In[190]:

np.where(summary_loss1tr.argsort() == 81)


# In[197]:

np.where(summary_loss1tr[5].argsort()==80)


# In[201]:

summary_loss1tr[5].argmax()


# In[208]:

summary_loss1tr.loc['SOUTH MELBOURNE']


# In[219]:

np.where(summary_loss1tr[5]>0.2)


# In[210]:

summary_loss1tr.ix[65]


# In[207]:

summary_loss1tr.ix[26] # ['ELWOOD']


# In[198]:

summary_loss1tr.ix[68]


# In[171]:

summary_loss1tr.loc['NIDDRIE']


# In[ ]:



