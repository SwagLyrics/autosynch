import os
import tarfile
import yaml

from scripts.training import training_process

root_dir = '/Users/Chris/autosynch'
tar_path = os.path.join(root_dir, 'resources/MedleyDB_V1.tar.gz')
metadata_dir = os.path.join(root_dir, 'resources/Metadata')
mixtures_dir = os.path.join(root_dir, 'src/mad_twinnet_train/dataset/mixtures')
sources_dir = os.path.join(root_dir, 'src/mad_twinnet_train/dataset/sources')

skip = dict.fromkeys(os.listdir(mixtures_dir))

with tarfile.open(tar_path) as tar:
    for file in os.listdir(metadata_dir):
        print(file)
        file = os.path.join(metadata_dir, file)

        with open(file, 'r') as f:
            metadata = yaml.safe_load(f)
        print('yaml loaded')

        if metadata['has_bleed'] == 'no' and metadata['instrumental'] == 'no':
            stem_dir = metadata['stem_dir']
            base_dir = os.path.join('V1', stem_dir[:-6])
            mix_filename = metadata['mix_filename']

            if mix_filename in skip:
                continue

            for stem in metadata['stems']:
                if metadata['stems'][stem]['component'] == 'melody' or 'rapper' in metadata['stems'][stem]['instrument']:
                    vox_filename = metadata['stems'][stem]['filename']
                    break

            mix_path = os.path.join(base_dir, mix_filename)
            vox_path = os.path.join(os.path.join(base_dir, stem_dir), vox_filename)

            print('extracting')
            mix_file = tar.getmember(mix_path)
            vox_file = tar.getmember(vox_path)
            mix_file.name = os.path.basename(mix_path)
            vox_file.name = os.path.basename(vox_path)

            tar.extract(mix_file, mixtures_dir)
            tar.extract(vox_file, sources_dir)

            print('done')

training_process()
