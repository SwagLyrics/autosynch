import os
import sys
import warnings
import logging

from autosynch.snd import SND
from autosynch.config import resources_dir
from autosynch.mad_twinnet.scripts import twinnet

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S')

def get_vocal_syllables(source, output_dir=None, get_background=False):
    """
    Performs MaD TwinNet and returns number of syllables in the vocals. Creates
    isolated vocals in 16-bit wav files.

    :param source: File name or list of file names to be processed.
    :type source: str | list[str]
    :param output_dir: Directory to put output files.\
           If None, outputs to directory of first file in source.
    :type output_dir: str
    :param get_background: Flag to output isolated background.
    :type get_background: bool
    :return snd_output: Number of syllables for each file in source.\
            Index for each syllable count matches index of source.
    :rtype: list[int | None]
    """

    logging.info('Initializing evaluation mechanism...')

    # Checks for valid source type
    if isinstance(source, str):
        source = [source]
    elif not isinstance(source, list):
        logging.error('Source must be either single file or list of files')
        raise TypeError('Incorrect input type')

    # Gets proper target names
    output_file_names = twinnet._make_target_file_names(source)
    if output_dir is None:
        output_dir = os.path.dirname(source[0])
    else:
        if not os.path.exists(output_dir):
            logging.error('%s must already exist', output_dir)
            sys.exit()
        for index, file in enumerate(output_file_names):
            f_name_0 = os.path.basename(file[0])
            f_name_1 = os.path.basename(file[1])
            file[0] = os.path.join(output_dir, f_name_0)
            file[1] = os.path.join(output_dir, f_name_1)

    logging.info('Evaluation mechanism successfully initialized')

    # MaD TwinNet
    twinnet.twinnet_process(source, output_file_names=output_file_names, get_background=get_background)

    # SND on each output vocal file
    logging.info('Beginning syllable nuclei detection...')
    snd = SND()
    snd_output = []
    for files in output_file_names:
        try:
            syllables = int(snd.run(files[0]))
            snd_output.append(syllables)
            logging.info('%s: %s syllables', files[0], syllables)
        except Exception as e:
            logging.error('SND failed for %s', files[0], exc_info=True)
            snd_output.append(None)

    return snd_output

def eval_by_syllable(source, vocals, output_dir=None, get_background=False):
    """
    Compares SND from MaD TwinNet versus original vocal source file.

    :param source: File name or list of file names to be processed.
    :type source: str | list[str]
    :param vocals: File name or list of file names of original vocal source files.
           Indices should correspond to files in 'source'.
    :param output_dir: Directory to put output files.\
           If None, outputs to directory of first file in source.
    :type output_dir: str
    :param get_background: Flag to output isolated background.
    :type get_background: bool
    :return eval_output: Comparison of SND results for MaD and original vocal source.
    :rtype: list[dict{ 'name': file_name,
                       'mad': num_syllables_mad_result,
                       'src': num_syllables_src_result,
                       'off': difference_between_mad_and_src_results
                      }]
    """

    eval_output = []
    snd = SND()

    if isinstance(source, str):
        source = [source]
    if isinstance(vocals, str):
        vocals = [vocals]
    if len(source) != len(vocals):
        raise IndexError('"source" must be same length as "vocals"')

    mad_syllables = get_vocal_syllables(source, output_dir, get_background)

    for index, syllables in enumerate(mad_syllables):
        if syllables is None:
            warnings.warn('No syllable data for {}'.format(source[index]), RuntimeWarning)
            continue
        src_syllables = int(snd.run(vocals[index]))
        eval_item = { 'name': source[index],
                      'mad': syllables,
                      'src': src_syllables,
                      'off': syllables-src_syllables
                    }
        eval_output.append(eval_item)

    return eval_output

if __name__ == '__main__':
    source = [ os.path.join(resources_dir, 'examples/LizNelson_Rainfall_MIX.wav'),
               os.path.join(resources_dir, 'examples/MarvinGaye_Grapevine_MIX.wav') ]
    vocals = [ os.path.join(resources_dir, 'examples/LizNelson_Rainfall_VOCALS.wav'),
               os.path.join(resources_dir, 'examples/MarvinGaye_Grapevine_VOCALS.wav') ]
    output_dir = os.path.join(resources_dir, 'outputs')

    eval = eval_by_syllable(source, vocals, output_dir, False)
    for item in eval:
        print()
        print(os.path.basename(item['name']))
        print('---------------------------------')
        print('Syllables from MaD TwinNet : {}'.format(item['mad']))
        print('Syllables from vocal source: {}'.format(item['src']))
        print('Number of syllables error  : {}'.format(item['off']))
