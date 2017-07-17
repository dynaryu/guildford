
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np


# In[3]:

from scipy import stats


# In[4]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 6.0)
pylab.rcParams['figure.dpi'] = 300
pylab.rcParams['font.size'] = 12
pylab.rcParams['legend.numpoints'] = 1


# In[4]:

dat = pd.read_excel('./Copy of Bridge Data for Earthquake Scenario GA modified.xlsx')


# In[5]:

dat.columns


# In[6]:

dat.shape


# In[7]:

sum(dat['HAZUS Class'].notnull())


# In[ ]:

dat.loc[dat['HAZUS Class'].notnull(), ['Latitude','Longitude','HAZUS Class','Skew angle','No. Of Spans']].to_csv('./aa.csv', index=True)


# In[ ]:

# appended by SITE_CLASS


# In[1]:

sel = pd.read_csv('./input/bridgedb_mel.csv')


# In[ ]:

sel.shape


# In[ ]:

sel.columns


# In[ ]:

np.where(sel['SPAN'].isnull())


# In[ ]:

import numpy as np


# In[38]:

sel = pd.read_csv('./input/bridgedb_mel.csv')


# In[39]:

sel.shape


# In[40]:

sel['STRUCTURE_CLASSIFICATION'].value_counts()


# In[41]:

sel.columns


# In[42]:

sel['SPAN'].notnull().sum()


# ##### sel['SITE_CLASS'].notnull().sum()

# In[5]:

import numpy as np


# In[46]:

tf_idx = np.where(sel['SPAN'].isnull())[0]
tf_ = sel.loc[sel['SPAN'].isnull(),'BID']


# In[47]:

tf_idx


# In[48]:

tf_


# In[49]:

dat.loc[tf_, 'No. Of Spans']


# In[50]:

dat.index


# In[52]:

sel.index


# In[54]:

sel.loc[205, 'LATITUDE'], sel.loc[205, 'LONGITUDE']


# In[57]:

sel.loc[tf_idx, 'SPAN'] = dat.loc[tf_, 'No. Of Spans'].values


# In[59]:

sel.loc[tf_idx, 'SPAN']


# In[62]:

sel['SITE_CLASS'].unique()


# In[63]:

sel['SITE_CLASS'].notnull().sum()


# In[72]:

sel.loc[sel['SITE_CLASS'].isnull(), 'BID']


# In[65]:

sel.loc[120, 'LATITUDE'], sel.loc[120, 'LONGITUDE']


# In[66]:

sel.head()


# In[68]:

sel['SPAN'] = sel['SPAN'].astype(int)


# In[71]:

sel.to_csv('./input/bridgedb_mel2.csv', index=False)


# In[1]:

# sel column order


# In[3]:

sel = pd.read_csv('./input/bridgedb_mel.csv')


# In[4]:

sel.columns


# In[5]:

column_list = ['BID','LATITUDE','LONGITUDE','STRUCTURE_CLASSIFICATION','STRUCTURE_CATEGORY','SKEW','SPAN','SITE_CLASS']


# In[7]:

sel[column_list].to_csv('./input/bridgedb_mel2.csv', index=False)


# In[1]:

# example


# In[2]:

sa03, sa10 = 1.5659359, 0.67504851


# In[5]:

import numpy as np


# In[169]:

gm = np.load('./output/mel_motion/soil_SA.npy')


# In[171]:

gm.shape


# In[228]:

id_gm = 56


# In[229]:

sa03 = gm[0, 0, 0, id_gm, 0, 1]


# In[66]:

sa03 = 2.1


# In[230]:

sa10 = gm[0, 0, 0, id_gm, 0, 2]


# In[67]:

sa10 = 0.43


# In[231]:

sa03, sa10


# In[9]:

import pandas as pd


# In[6]:

site = pd.read_csv('./input/bridgedb_mel.csv')


# In[7]:

site.head()


# In[232]:

site.iloc[id_gm]


# ###### bridge_type, _, skew, no_span, site_class = 'HWB22','BRIDGE',15,1,'C'

# In[178]:

K3D, Ishape = 'EQ2', 


# In[234]:

K3D, Ishape = 'EQ4', 0 # RLB10 == HWB12


# In[179]:

A, B=0.33, 0.0 # EQ2


# In[71]:

A, B = 0.25, 1


# In[235]:

A, B = 0.09, 1 # EQ4


# In[237]:

try:
    K3d = 1+A/(no_span-B)
except ZeroDivisionError:
    K3d = 1+A


# In[238]:

Kskew = np.sqrt(np.sin(np.deg2rad(90.0-skew)))


# In[181]:

Kskew = 1.0


# In[182]:

np.sin(np.deg2rad(90.0))


# In[240]:

if Ishape > 0:
    Kshape = 2.5*sa10/sa03
else:
    Kshape  = 1.0


# In[184]:

med = [0.6, 0.9, 1.1, 1.5] # HWB22


# In[241]:

med = [0.25, 0.35, 0.45, 0.70]


# In[74]:

med = [0.25, 0.35, 0.45, 0.70]


# In[242]:

A2 = Kshape*med[0]


# In[243]:

A3 = Kskew*K3d*med[1]


# In[244]:

A4 = Kskew*K3d*med[2]


# In[245]:

A5 = Kskew*K3d*med[3]


# In[246]:

Kshape, Kskew, K3d


# In[247]:

A2, A3, A4, A5


# In[161]:

from scipy import stats


# In[254]:

x_ = np.arange(0, 1.0, 0.05)


# In[259]:

stats.lognorm.cdf(0.1, 0.6, scale=0.25)


# In[258]:

plt.plot(x_, stats.lognorm.cdf(x_, 0.6, scale=0.25))


# In[248]:

a = np.ones((5))
a[1] = stats.lognorm.cdf(sa10, 0.6, scale=A2)


# In[249]:

a[2] = stats.lognorm.cdf(sa10, 0.6, scale=A3)


# In[250]:

a[3] = stats.lognorm.cdf(sa10, 0.6, scale=A4)


# In[251]:

a[4] = stats.lognorm.cdf(sa10, 0.6, scale=A5)


# In[252]:

a


# In[196]:

np.hstack((-1.0*np.diff(a), a[4]))


# In[78]:

damage = pd.read_csv('./output/mel_structural_damage.txt', names = ['id', 'slight', 'moderate', 'extensive', 'complete'], skiprows=1)


# In[79]:

damage.head()


# In[80]:

damage.dtypes


# In[81]:

damage.columns


# In[82]:

damage['slight'].max()


# In[202]:

gm.shape


# In[203]:

gm[0, 0, 0, :, 0, 2].max()


# In[204]:

from eqrm_code.RSA2MMI import rsa2mmi_array


# In[205]:

rsa2mmi_array([0.1], period=1.0)


# In[206]:

site.columns


# In[11]:

site['STRUCTURE_CLASSIFICATION'].value_counts()


# In[216]:

idx = np.where(site['STRUCTURE_CLASSIFICATION']=='RLB10')


# In[223]:

gm[0, 0, 0, idx[0], 0, 2]


# In[222]:

idx[0][5]


# In[226]:

gm[0, 0, 0, 56, 0, :]


# In[ ]:

gm[0, 0, 0, 56]


# In[275]:

(damage['slight'] + damage['moderate'] + damage['extensive']+damage['complete']).plot()


# In[1]:

# estimate number of bridge in damage state


# In[6]:

damage = pd.read_csv('./output/mel_structural_damage.txt', names = ['id', 'slight', 'moderate', 'extensive', 'complete'], skiprows=1)


# In[7]:

damage.columns


# In[11]:

damage.shape


# In[83]:

np.random.seed(99)


# In[85]:

nsample = 1000
sampled = np.random.uniform(size=(damage.shape[0], nsample))


# In[86]:

sampled.shape


# In[87]:

damage_pe = np.zeros(shape=(damage.shape[0],4) )


# In[88]:

damage_pe.shape


# In[89]:

damage_pe[:, 3] = 1 - damage['complete'].values
damage_pe[:, 2] = damage_pe[:, 3] - damage['extensive'].values
damage_pe[:, 1] = damage_pe[:, 2] - damage['moderate'].values
damage_pe[:, 0] = damage_pe[:, 1] - damage['slight'].values



# In[90]:

damage.loc[56]


# In[91]:

damage_pe[56, :]


# In[92]:

damage_pe[0, :]


# In[93]:

df_ds = pd.DataFrame(0, index=range(damage.shape[0]), columns=range(nsample))
for irow in range(damage.shape[0]):
    df_ds.loc[irow, :] = np.digitize(sampled[irow, :], damage_pe[irow, :])
    
#np.digitize(sampled, damage_pe)


# In[94]:

df_ds.columns


# In[96]:

df_ds.shape


# In[115]:

print damage.loc[11], df_ds.loc[11].value_counts()/1000.0


# In[117]:

damage_pe[11,:]


# In[118]:




# In[106]:

damage_pe[11,:]


# In[72]:

# df_ds['class'] = site['STRUCTURE_CLASSIFICATION']


# In[70]:

df_ds.apply(pd.value_counts, axis=0)


# In[79]:

gr1 = df_ds['class'].apply(lambda x: 'RAIL' if 'RLB' in x else 'HIGH') 


# In[80]:

gr1.value_counts()


# In[101]:

df_ds['broad'] = gr1


# In[84]:

df_ds.drop('broad', 1)
df_ds.drop


# In[105]:

for i, grouped in df_ds.groupby('broad'):
    print i, grouped.apply(pd.value_counts, axis=0).fillna(0.0).mean(axis=1)


# In[31]:

perc = np.arange(5, 100, 5)

days = pd.DataFrame(0, index=damage.index, columns = perc)

for x in perc:
    file_ = './output/mel_bridge_days_to_complete_fp[{}].csv'.format(x)
    a = pd.read_csv(file_, skiprows=2, names=['a', 'b', 'c'])
    days[x] = a['a']


# In[41]:

damage['complete'].argmax()


# In[42]:

damage.loc[247]


# In[43]:

damage['slight'].argmax()


# In[45]:


damage.loc[143]


# In[52]:

days[95].argmax()


# In[53]:

days.loc[56]


# In[54]:

damage.loc[56]


# In[55]:

from eqrm_code import bridge_time_complete 


# In[67]:

fp = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
states = np.array([[[1],[0],[3]]])
#states = np.array([[[1],[0],[3]],
#                    [[2],[1],[4]]])
bridge_time_complete.time_to_complete(fp, states)


# In[77]:

stats.norm.ppf(0.1, 75, 42)


# In[64]:

bridge_time_complete.time_to_complete(np.array([5, 10, 95]), np.array([[[0], [1], [2], [3], [4]]]))


# In[ ]:



