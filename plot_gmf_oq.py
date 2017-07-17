import pandas as pd
import os
import sys
import numpy as np

from plot_gmfield import plot_gmfield


def main():

    gm_path = sys.argv[1]
    real_num = sys.argv[2]

    _file = os.path.join(gm_path, 'realizations_{}.csv'.format(real_num)) 
    tmp = pd.read_csv(_file)
    gsim_weights = tmp[['gsim', 'weight']].to_dict('list')

    periods = ['PGA', 'SA(0.3)', 'SA(1.0)']

    range_dic = {'MMI': np.arange(4.0, 8.5, 0.5),
                 'PGA': np.arange(0, 0.4, 0.025),
                 'SA(0.3)': np.arange(0.0, 0.5, 0.1),
                 'SA(1.0)': np.arange(0, 0.11, 0.02)}


    # for period in periods:
    for period in ['PGA']:

        combined = []
        for _gsim, _weight in zip(gsim_weights['gsim'], gsim_weights['weight']):

            _file = 'gmf-{}-{}_{}.csv'.format(_gsim, period, real_num)

            gm = pd.read_csv(os.path.join(gm_path, _file))

            _png_file = os.path.join(gm_path, 'gmf_{}_{}_{}.png'.format(period, _gsim, real_num))
            plot_gmfield(gm['lat'].values, gm['lon'].values, gm['000'].values, range_dic[period], _png_file)

            combined.append(gm['000'] * _weight)

        _array = np.array(combined).sum(axis=0)

        _png_file = os.path.join(gm_path, 'gmf_{}_{}.png'.format(period, real_num))
        plot_gmfield(gm['lat'].values, gm['lon'].values, _array, range_dic[period], _png_file)


if __name__ == '__main__':
    main()

