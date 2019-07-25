#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Usage script.
"""

from __future__ import print_function

import os
import sys
import time
import logging

import numpy as np
import torch

from ..modules import MaD
from ..helpers.data_feeder import data_feeder_testing, data_process_results_testing
from ..helpers.settings import debug, hyper_parameters, output_states_path, \
       training_constants, usage_output_string_per_example, usage_output_string_total

__author__ = ['Konstantinos Drossos -- TUT', 'Stylianos Mimilakis -- Fraunhofer IDMT']
__docformat__ = 'reStructuredText'
__all__ = ['twinnet_process']


def _make_target_file_names(sources_list):
    """Makes the target file names for the sources list.

    :param sources_list: The sources list.
    :type sources_list: list[str]
    :return: The target names.
    :rtype: list[list[str]]
    """
    targets_list = []

    for source in sources_list:
        f_name = os.path.splitext(source)[0]
        targets_list.append(['{}_voice.wav'.format(f_name), '{}_bg_music.wav'.format(f_name)])

    return targets_list


def twinnet_process(sources_list, output_file_names=None, get_background=False):
    """Applies MaD TwinNet to audio files.

    :param sources_list: Files to be processed.\
           Each must be .wav with 44.1kHz sample rate, recommended bit width of
           16 (8, 24, 32 acceptable).
    :type sources_list: str | list[str]
    :param output_file_names: The output file names to be used.\
           If None, outputs names are automatically generated.
    :type output_file_names: list[list[str]] | None
    :return: Whether or not TwinNet was successful
    :rtype: bool
    """

    logging.info('Initializing MaD TwinNet...')

    if isinstance(sources_list, str):
        sources_list = [sources_list]
    if output_file_names is None:
        output_file_names = _make_target_file_names(sources_list)
    if len(sources_list) != len(output_file_names):
        logging.error('Number of sources and number of outputs do not match')
        return False

    device = 'cuda' if not debug and torch.cuda.is_available() else 'cpu'
    logging.info('Device set as %s', device)

    # Masker modules
    mad = MaD(
        rnn_enc_input_dim=hyper_parameters['reduced_dim'],
        rnn_dec_input_dim=hyper_parameters['rnn_enc_output_dim'],
        original_input_dim=hyper_parameters['original_input_dim'],
        context_length=hyper_parameters['context_length'])

    mad.load_state_dict(torch.load(output_states_path['mad']))
    mad = mad.to(device).eval()

    # Data iterator
    testing_it = data_feeder_testing(
        window_size=hyper_parameters['window_size'], fft_size=hyper_parameters['fft_size'],
        hop_size=hyper_parameters['hop_size'], seq_length=hyper_parameters['seq_length'],
        context_length=hyper_parameters['context_length'], batch_size=1,
        debug=debug, sources_list=sources_list
    )

    logging.info('MaD TwinNet successfully initialized')

    logging.info('Beginning vocal isolation...')
    total_time = 0

    for index, data in enumerate(testing_it()):

        s_time = time.time()

        mix, mix_magnitude, mix_phase, voice_true, bg_true = data

        voice_predicted = np.zeros((
                mix_magnitude.shape[0],
                hyper_parameters['seq_length'] - hyper_parameters['context_length'] * 2,
                hyper_parameters['window_size']), dtype=np.float32)

        logging.info('Applying MaD TwinNet per batch...')
        n_batches = int(mix_magnitude.shape[0] / training_constants['batch_size'])
        for batch in range(n_batches):
            b_start = batch * training_constants['batch_size']
            b_end = (batch + 1) * training_constants['batch_size']

            v_in = torch.from_numpy(
                mix_magnitude[b_start:b_end, :, :]).to(device)

            voice_predicted[b_start:b_end, :, :] = mad(
                v_in).v_j_filt.cpu().detach().numpy()

            logging.info('Batch %d/%d complete', batch+1, n_batches)

        logging.info('Writing output to file...')
        data_process_results_testing(
            index=index, voice_true=voice_true, bg_true=bg_true,
            voice_predicted=voice_predicted,
            window_size=hyper_parameters['window_size'], mix=mix, mix_magnitude=mix_magnitude,
            mix_phase=mix_phase, hop=hyper_parameters['hop_size'],
            context_length=hyper_parameters['context_length'],
            output_file_name=output_file_names[index],
            get_background=get_background
        )

        e_time = time.time()

        logging.info(usage_output_string_per_example.format(
            f=sources_list[index],
            t=e_time - s_time))

        total_time += e_time - s_time

    logging.info(usage_output_string_total.format(t=total_time))
    logging.info('MaD TwinNet completed')

    return True

# EOF
