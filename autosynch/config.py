import os

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
resources_dir = os.path.join(parent_dir, 'resources')
tests_dir = os.path.join(parent_dir, 'tests')
align_tests_dir = os.path.join(resources_dir, 'align_tests')

praat_script_path = os.path.join(resources_dir, 'praat-script.txt')
cmudict_path = os.path.join(resources_dir, 'cmudict.0.7a.txt')
nettalk_path = os.path.join(resources_dir, 'nettalk.data')
