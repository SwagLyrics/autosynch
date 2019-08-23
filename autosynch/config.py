import os

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
resources_dir = os.path.join(parent_dir, 'resources')
tests_dir = os.path.join(parent_dir, 'tests')
evals_dir = os.path.join(parent_dir, 'optimization_evals')
align_tests_dir = os.path.join(resources_dir, 'align_tests')

praat_script_path = os.path.join(resources_dir, 'praat-script.txt')
cmudict_path = os.path.join(resources_dir, 'cmudict.0.7a.txt')
nettalk_path = os.path.join(resources_dir, 'nettalk.data')

dp_err_matrix = {'chorus': {'chorus': 2, 'verse': 4, 'other': 9},
                 'verse':  {'chorus': 4, 'verse': 3, 'other': 8},
                 'intro':  {'chorus': 6, 'verse': 6, 'other': 5},
                 'bridge': {'chorus': 8, 'verse': 8, 'other': 6}}
