"""
  EQRM parameter file
  All input files are first searched for in the input_dir, then in the
  resources/data directory, which is part of EQRM.

 All distances are in kilometers.
 Acceleration values are in g.
 Angles, latitude and longitude are in decimal degrees.

 If a field is not used, set the value to None.


"""

from os.path import join, expanduser
from eqrm_code.parse_in_parameters import eqrm_data_home, get_time_user
from numpy import arange

# Path
working_path = join(expanduser("~"),'Projects/scenario_Guildford')

# Operation Mode
run_type = "bridge"
is_scenario = True
site_tag = "perth"
site_db_tag = ""
# return_periods = [10, 50, 100, 200, 250, 474.56, 500, 974.78999999999996, 1000, 2474.9000000000001, 2500, 5000, 7500, 10000]
return_periods = [10]
input_dir = join(working_path, 'input')
output_dir = join(working_path, 'bridge_Mw5.6D7')
# use_site_indexes = True
use_site_indexes = False
del working_path
#site_indexes = [1, 2]
zone_source_tag = ""
event_control_tag = ""

# Scenario input
scenario_magnitude = 5.6
scenario_depth = 7.0
scenario_azimuth = 0
scenario_dip = 60
scenario_latitude = -31.890
scenario_longitude = 115.994
scenario_number_of_events = 1

# Probabilistic input

# Attenuation
atten_models = ['Akkar_2010_crustal','Campbell08','Atkinson06_bc_boundary_bedrock']
atten_model_weights = [0.3, 0.4, 0.3]
#atten_models = ['Atkinson06_bc_boundary_bedrock']
#atten_model_weights = [1.0]
atten_collapse_Sa_of_atten_models = True
atten_periods = [0.0, 0.3, 1.0] #hyeuk
atten_variability_method = None
atten_threshold_distance = 400
atten_override_RSA_shape = None
atten_cutoff_max_spectral_displacement = False
atten_pga_scaling_cutoff = 10.0
atten_smooth_spectral_acceleration = None
atten_log_sigma_eq_weight = 0

# Amplification
use_amplification = True
amp_variability_method = None
amp_min_factor = 0.6
amp_max_factor = 10000

# Buildings
# bridges_functional_percentages = arange(5, 100, 5)
bridges_functional_percentages = None

# Capacity Spectrum Method

# Loss

# Save
save_hazard_map = False
save_motion = True
save_prob_structural_damage = True

# If this file is executed the simulation will start.
# Delete all variables that are not EQRM parameters variables.
if __name__ == '__main__':
    from eqrm_code.analysis import main
    main(locals())
