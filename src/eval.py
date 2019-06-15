import os
import sys
import logging

import config, snd
from mad_twinnet.scripts import twinnet

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S')

def get_vocal_syllables(source, output_dir=None, get_background=False):

    logging.info('Initializing evaluation mechanism...')
    if isinstance(source, str):
        source = [source]
    elif not isinstance(source, list):
        logging.error('Source must be either single file path or list of file paths')
        sys.exit()

    if output_dir is None:
        output_file_names = None
    else:
        if not os.path.exists(output_dir):
            logging.error('%s must already exist', output_dir)
            sys.exit()
        output_file_names = twinnet._make_target_file_names(source)
        for index, file in enumerate(output_file_names):
            f_name_0 = os.path.basename(file[0])
            f_name_1 = os.path.basename(file[1])
            file[0] = os.path.join(output_dir, f_name_0)
            file[1] = os.path.join(output_dir, f_name_1)

    logging.info('Evaluation mechanism successfully initialized')

    twinnet.twinnet_process(source, output_file_names=output_file_names, get_background=get_background)

    logging.info('Beginning syllable nuclei analysis...')
    for file in output_dir:
        f_name = os.path.basename(file)
        syllables = snd.snd(file)
        logging.info('%s: %d syllables', f_name, syllables)

def eval_by_syllable():
    pass

get_vocal_syllables('/Users/Chris/autosynch/resources/MedleyDB_sample/Audio/LizNelson_Rainfall/LizNelson_Rainfall_MIX.wav',
                    '/Users/Chris/autosynch/resources/output')
