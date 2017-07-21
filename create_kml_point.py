# a scrip to make contour map and KML for google earth

import os
import pandas as pd
import simplekml


#df = pd.read_csv('./input/Guildford_res.csv')

df = pd.read_csv('../bridge_Mw5.6D7/result_pe.csv')
icon_scale_value  = 0.70

# top 10
id_top10 = df['pe_slight'].argsort()[::-1][:10].tolist()

# point kml
kml = simplekml.Kml()

schema = kml.newschema()
list_ = ['BID', 'STRUCTURE_CLASSIFICATION', 'SPAN', 'SKEW', 'SA03', 'SA10', 'pe_slight', 'pe_moderate', 'pe_extensive', 'pe_complete']

to_float = ['SKEW', 'SA03', 'SA10', 'pe_slight', 'pe_moderate', 'pe_extensive', 'pe_complete']

for item in list_:
    if item in to_float:
       schema.newgxsimplearrayfield(name=item,
                                    type=simplekml.Types.float,
                                    displayname=item)
    else:
        schema.newgxsimplearrayfield(name=item,
                                     type=simplekml.Types.string,
                                     displayname=item)

style1 = simplekml.Style()
style1.labelstyle.color = 00000000  # Make the text red
style1.labelstyle.scale = 0.000000  # Make the text twice as big
style1.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/target.png'
style1.iconstyle.scale = icon_scale_value

style2 = simplekml.Style()
style1.labelstyle.color = 00000000  # Make the text red
style2.labelstyle.scale = 0.000000  # Make the text twice as big
style2.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/target.png'
style2.iconstyle.scale = icon_scale_value
style2.iconstyle.color = 'ff00ffff'


for irow, row in df.iterrows():  # Generate latitude values

    pnt = kml.newpoint(name='{}'.format(irow))
    pnt.coords = [(row['LONGITUDE'], row['LATITUDE'])]

    if irow in id_top10:
        pnt.style = style2
    else:
        pnt.style = style1

    #for item, flag in zip(list_, row.notnull()):
    #    if flag:
    for item in list_:

        if item in to_float:
            pnt.extendeddata.schemadata.newsimpledata(item, '{:.2f}'.format(row[item])) # Ditto
        else:
            pnt.extendeddata.schemadata.newsimpledata(item, row[item]) # Ditto

output_path = './'
#output_file = os.path.join(output_path, 'residential.kml')
output_file = os.path.join(output_path, 'bridge_damage.kml')
kml.save(output_file)
print '{} is generated'.format(output_file)

###############################################################################
# main program

#csv_file = sys.argv[1]
#output_path = os.path.dirname(os.path.abspath(csv_file))

# read ground motion grid data
#df = pd.read_csv(csv_file)

# 5 levels
# icon_list = ['Layer0_Symbol_1c9ac690_0.png',
#              'Layer0_Symbol_1c9acde8_0.png',
#              'Layer0_Symbol_1c9acc70_0.png',
#              'Layer0_Symbol_1c9acaf8_0.png',
#              'Layer0_Symbol_1c9ac980_0.png']

# 6 levels
# icon_list = ['Layer0_Symbol_1c9ac690_0.png',
#              'Layer0_Symbol_1c9acde8_0.png',
#              'Layer0_Symbol_1c9acc70_0.png',
#              'Layer0_Symbol_1c9acaf8_0.png',
#              'Layer0_Symbol_1c9ac980_0.png',
#              'Layer0_Symbol_1c9ac980_1.png']

# epicentre_dic = {'FLAG': 1, 'LAT': -38.36, 'LON': 146.65}
# icon_scale_value = 0.375

# if len(sys.argv) < 3:
#     print df.min(axis=0)
#     print df.max(axis=0)

# else:
#     key_str = sys.argv[2]
#     bins = [float(x) for x in sys.argv[3].strip().split(',')]
#     create_kml_point(df, key_str, bins, icon_list, icon_scale_value,
#                      epicentre_dic, output_path)
