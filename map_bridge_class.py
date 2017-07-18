import pandas as pd

# read bridge data
bridge = pd.read_excel('./input/Guildford%20study%20bridge%20exposure.xlsx', 
	sheetname='Bridges_Guildford_study')

"""
bridge.columns
Out[14]: 
Index([u'Strucutre No', u'Lattitude', u'Longitude', u'Road Name',
       u'Crossine Name', u'Local Government', u'Bridge Owner', u'Function',
       u'Usage', u'Commonwealth Class Name', u'Functional Class Name',
       u'Link Subcategory Name', u'Bridge_Age', u'STR_TYPE_NAME',
       u'SUPERSTRUCTURE_NAME', u'Spans', u'Skew Angle', u'Pier Type',
       u'Pier - Deck Connection', u'Support Description',
       u'Earthquake design?', u'Overall Length', u'Longest Span',
       u'HAZUS Rail Bridge Class'],
      dtype='object')
"""

# fill NaN with ""
bridge.fillna('',inplace=True)

# assign design level
# =< 13: code, > 13: pre_code
bridge['Earthquake design'] = bridge['Bridge_Age'].apply(lambda x: True if x <= 13 else False)

# re_assign STR_TYPE_NAME
# Timber Hybrid == Timber
# Steel/Concrete Composite == Steel
# Tunnel 
"""
In [48]: bridge['STR_TYPE_NAME'].value_counts()
Out[48]: 
Prestressed Concrete        61
Reinforced Concrete         26
Timber                      14
Steel/Concrete Composite     6
Steel                        6
Timber Hybrid                2
Tunnel                       1
"""

def mapping_hazus_class(row):


	def assign_by_design(value_pre, value_code):
		if row['Earthquake design']:
			return value_code
		else:
			return value_pre

	def assign_steel(row):
		if row['Overall Length'] < 20:
			if 'Simply' in row['Support Description']:
				return 24
			elif 'Continuous' in row['Support Description']:
				return 26
			else:
				print('WARNING: Support Description is invalid: {}'.format(row['Strucutre No']))
		else:
			if 'Simply' in row['Support Description']:
				return assign_by_design(12, 14)
			elif 'Continuous' in row['Support Description']:
				return assign_by_design(15, 16)
			else:
				print('WARNING: Support Description is invalid: {}'.format(row['Strucutre No']))


	def assign_rc(row):
		if 'Simply' in row['Support Description']:
			return assign_by_design(5, 7)

		elif 'Continuous' in row['Support Description']:
			if ('BOX' in row['SUPERSTRUCTURE_NAME']) and ('Pier Type' == '1 Column'):
				return 8
			else:
				return assign_by_design(10, 11)
		else:
			print('WARNING: Support Description is invalid: {}'.format(row['Strucutre No']))

	def assign_pc(row):
		if 'Simply' in row['Support Description']:
			return assign_by_design(17, 19)

		elif 'Continuous' in row['Support Description']:
			if ('BOX' in row['SUPERSTRUCTURE_NAME']) and ('Pier Type' == '1 Column'):
				return 20
			else:
				return assign_by_design(22, 23)
		else:
			print('WARNING: Support Description is invalid: {}'.format(row['Strucutre No']))


	if row['HAZUS Rail Bridge Class']:
		return row['HAZUS Rail Bridge Class']
	elif 'Tunnel' in row['STR_TYPE_NAME']:
		return 'RTU2'
	else:
		if row['Spans'] < 2:
			value = assign_by_design(3, 4)
		else:
			if row['Longest Span'] > 150.0:
				value = assign_by_design(1, 2)
			else:
				if 'Prestressed' in row['STR_TYPE_NAME']:
					value = assign_pc(row)
				elif 'Reinforced' in row['STR_TYPE_NAME']:
					value = assign_rc(row)
				elif 'Steel' in row['STR_TYPE_NAME']:
					value = assign_steel(row)
				else:
					value = 28

		return 'HWB{}'.format(value)

bridge['HAZUS_Road_Bridge_Class'] = bridge.apply(mapping_hazus_class, axis=1)  

# create bridgedb_perth.csv
# BID,LATITUDE,LONGITUDE,STRUCTURE_CLASSIFICATION,STRUCTURE_CATEGORY,SKEW,SPAN,SITE_CLASS

a = pd.read_csv('../input/Guildford_study_bridge_exposure.csv')
a['STRUCTURE_CATEGORY'] = 'BRIDGE'

site_class = pd.read_csv('../input/XYbridge_par_siteclass.csv')

id_to_site_class = []
for _id, item in a.iterrows():
	lat, lon = item['Lattitude'], item['Longitude']

	idx = np.argmin((site_class['LATITUDE']-lat)**2 + (site_class['LONGITUDE']-lon)**2)

	id_to_site_class.append(idx)
	print('{}:{}'.format(lat, site_class.loc[idx, 'LATITUDE']))
	print('{}:{}'.format(lon, site_class.loc[idx, 'LONGITUDE']))


a['SITE_CLASS'] = site_class.loc[id_to_site_class, 'SITECLASS']

a[['Strucutre No', u'Lattitude', u'Longitude', u'HAZUS_Road_Bridge_Class', 'STRUCTURE_CATEGORY', u'Skew Angle', u'Spans', 'SITE_CLASS']].to_csv('../input/bridgedb_perth.csv')
