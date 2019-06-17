import os
import tarfile
import yaml

tar_path = '/home/cwang/MedleyDB_Sample.tar.gz'
mixtures_dir = '/home/cwang/mad_twinnet_train/dataset/mixtures'
sources_dir = '/home/cwang/mad_twinnet_train/dataset/sources'

with tarfile.open(tar_path) as tar:
    file = tar.next()
    while file is not None:
        basename = os.path.basename(file.name)
        if basename.endswith('.yaml') and not basename.startswith('._'):
            metadata = yaml.load(tar.extractfile(file).read().decode('utf-8'))
            if metadata['excerpt'] == 'no' and metadata['has_bleed'] == 'no' and metadata['instrumental'] == 'no':
                root_dir = os.path.dirname(file.name)
                stem_dir = metadata['stem_dir']
                mix_filename = metadata['mix_filename']
                for stem in metadata['stems']:
                    if metadata['stems'][stem]['component'] == 'melody':
                        vox_filename = metadata['stems'][stem]['filename']
                        break

                mix_path = os.path.join(root_dir, mix_filename)
                vox_path = os.path.join(os.path.join(root_dir, stem_dir), vox_filename)

                mix_file = tar.getmember(mix_path)
                vox_file = tar.getmember(vox_path)
                mix_file.name = os.path.basename(mix_path)
                vox_file.name = os.path.basename(vox_path)

                tar.extract(mix_file, mixtures_dir)
                tar.extract(vox_file, sources_dir)

        file = tar.next()
