#include <sndfile.h>

sf_count_t sfx_mix_mono_read(SNDFILE *file, float *data, sf_count_t len) {

    SF_INFO info;
    sf_command(file, SFC_GET_CURRENT_SF_INFO, &info, sizeof(info));

    if (info.channels == 1)
        return sf_read_float(file, data, len);

    static float multi_data[2048];
    int i, j, frames_read;
    sf_count_t dataout = 0;

    while (dataout < len) {

        int this_read = sizeof(multi_data)/sizeof(float)/info.channels;
        if (len < this_read)
            this_read = len;

        frames_read = sf_readf_float(file, multi_data, this_read);
        if (frames_read == 0)
            break;
        for (i = 0; i < frames_read; i++) {
            float mix = 0.0;
            for (j = 0; j < info.channels; j++)
                mix += multi_data[i*info.channels+j];
            data[dataout+i] = mix/info.channels;
        }

        dataout += this_read;
    }

    return dataout;
}

static void mix_to_mono(SNDFILE *input, SNDFILE *output) {

    float buf[1024];
    sf_count_t count;

    while ((count = sfx_mix_mono_read(input, buf, sizeof(buf)/sizeof(float))) > 0)
        sf_write_float(output, buf, count);
}

