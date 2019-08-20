#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The settings of the modules and the process
"""

import os

__author__ = 'Konstantinos Drossos'
__docformat__ = 'reStructuredText'
__all__ = [
    'debug',
    'dataset_paths',
    'output_audio_paths',
    'metrics_paths',
    'output_states_path',
    'training_output_string',
    'testing_output_string_per_example',
    'testing_output_string_all',
    'training_constants',
    'wav_quality',
    'hyper_parameters',
    'usage_output_string_per_example',
    'usage_output_string_total'
]


debug = False
_version_suffix = ''

# Paths
_root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
_dataset_parent_dir = os.path.join(_root_dir, 'dataset')
_outputs_path = os.path.join(_root_dir, 'outputs')
_states_path = os.path.join(_outputs_path, 'states')
_metrics_path = os.path.join(_outputs_path, 'metrics')
_audio_files_path = os.path.join(_outputs_path, 'audio_files')

dataset_paths = {
    'mixtures': os.path.join(_dataset_parent_dir, 'Mixtures'),
    'sources': os.path.join(_dataset_parent_dir, 'Sources')
}

output_audio_paths = {
    'voice_true': os.path.join(
        _audio_files_path,
        'test_example_{}_voice_true.wav'.format('{p:02d}')),
    'voice_predicted': os.path.join(
        _audio_files_path,
        'test_example_{}_voice_predicted.wav'.format('{p:02d}')),
    'bg_true': os.path.join(
        _audio_files_path,
        'test_example_{}_bg_true.wav'.format('{p:02d}')),
    'bg_predicted': os.path.join(
        _audio_files_path,
        'test_example_{}_bg_predicted.wav'.format('{p:02d}')),
    'mix': os.path.join(
        _audio_files_path,
        'test_example_{}_mix_true.wav'.format('{p:02d}'))
}

metrics_paths = {
    'sdr': os.path.join(_metrics_path, 'sdr_p2.pckl'),
    'sir': os.path.join(_metrics_path, 'sir_p2.pckl')
}

output_states_path = {
    'mad': os.path.join(_states_path, 'mad{}.pt'.format(_version_suffix))
}

# Strings
training_output_string = 'Epoch: {ep:3d} Losses: -- ' \
                         'Masker:{l_m:6.4f} | Denoiser:{l_d:6.4f} | ' \
                         'Twin:{l_tw:6.4f} | Twin reg.:{l_twin:6.4f} | ' \
                         'Time:{t:6.2f} sec(s)'

testing_output_string_per_example = 'Example: {e:2d}, Median -- ' \
                                    'SDR:{sdr:6.2f} dB | SIR:{sir:6.2f} dB | ' \
                                    'Time:{t:6.2f} sec(s)'

testing_output_string_all = 'Median SDR:{sdr:6.2f} dB | ' \
                            'Median SIR:{sir:6.2f} dB | ' \
                            'Total time:{t:6.2f} sec(s)'

usage_output_string_per_example = 'File {f} processed. Time: {t:6.2f} sec'
usage_output_string_total = 'All files processed. Total time: {t:6.2f} sec'

# Process constants
training_constants = {
    'epochs': 2 if debug else 100,
    'batch_size': 16,
    'files_per_pass': 4
}

wav_quality = {'sampling_rate': 44100, 'nb_bits': 16}

# Hyper-parameters
hyper_parameters = {
    'window_size': 2049,
    'fft_size': 4096,
    'hop_size': 384,
    'seq_length': 60,
    'context_length': 10,
    'reduced_dim': 744,
    'original_input_dim': 2049,
    'learning_rate': 1e-4,
    'max_grad_norm': .5,
    'lambda_l_twin': .5,
    'lambda_1': 1e-2,
    'lambda_2': 1e-4
}
hyper_parameters.update({
    'rnn_enc_output_dim': 2 * hyper_parameters['reduced_dim']
})

# EOF
