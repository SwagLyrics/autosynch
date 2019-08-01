import os

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
resources_dir = os.path.join(parent_dir, 'resources')
tests_dir = os.path.join(parent_dir, 'tests')
align_tests_dir = os.path.join(resources_dir, 'align_tests')

praat_script_path = os.path.join(resources_dir, 'praat-script.txt')
cmudict_path = os.path.join(resources_dir, 'cmudict.0.7a.txt')
nettalk_path = os.path.join(resources_dir, 'nettalk.data')

dp_err_matrix = {'chorus': {'chorus': 0, 'verse': 5, 'other': 7},
                 'verse':  {'chorus': 5, 'verse': 2, 'other': 5},
                 'intro':  {'chorus': 3, 'verse': 5, 'other': 3},
                 'bridge': {'chorus': 5, 'verse': 3, 'other': 3}}
