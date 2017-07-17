"""
  EQRM parameter file
All input files are first searched for in the input_dir,then in the
resources/data directory, which is part of EQRM.

All distances are in kilometers.
Acceleration values are in g.
Angles, latitude and longitude are in decimal degrees.

If a field is not used, set the value to None.
"""

from os.path import join
from eqrm_code.parse_in_parameters import eqrm_data_home, get_time_user

# Operation Mode
run_type = "hazard" 
is_scenario = True
max_width = 15
site_tag = "Guildford" 
site_db_tag = "" 
return_periods = [500]
input_dir = join('.', 'input')
atten_models = ['ACS'] 
attn = atten_models[0]
output_dir = join('.', 'output', attn )
output_dir = join('.', 'output', 'Guildford_56_07_ACA')
del attn
input_dir = join('.', 'input')
zone_source_tag = ""
event_control_tag = ""
use_site_indexes = False
site_indexes = [2997, 2657, 3004, 3500]

# Scenario input
scenario_magnitude = 5.6
scenario_depth = 7.0
scenario_azimuth = 0
scenario_dip = 60
scenario_latitude = -31.890
scenario_longitude = 115.994
scenario_number_of_events = 1

# Probabilistic input
atten_models = ['Akkar_2010_crustal','Campbell08','Atkinson06_bc_boundary_bedrock']
atten_model_weights = [0.3, 0.4, 0.3]
atten_collapse_Sa_of_atten_models = True
atten_variability_method = 1
atten_variability_method = 5
atten_periods = [0.01]
atten_threshold_distance = 400
atten_override_RSA_shape = None
atten_cutoff_max_spectral_displacement = False
atten_pga_scaling_cutoff = 4
atten_smooth_spectral_acceleration = None
atten_log_sigma_eq_weight = 0

# Amplification
use_amplification = True
amp_variability_method = None
amp_min_factor = 0.6
amp_max_factor = 10000

# Buildings

# Capacity Spectrum Method

# Loss

# Save
save_hazard_map = False
save_total_financial_loss = False
save_building_loss = False
save_contents_loss = False
save_motion = True
save_prob_structural_damage = None

file_array = False

# If this file is executed the simulation will start.
# Delete all variables that are not EQRM attributes variables. 
if __name__ == '__main__':
    from eqrm_code.analysis import main
    main(locals())
