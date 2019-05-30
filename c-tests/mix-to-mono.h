#ifndef __MIX_TO_MONO_H__
#define __MIX_TO_MONO_H__

sf_count_t sfx_mix_mono_read(SNDFILE *file, float *data, sf_count_t len);
static void mix_to_mono(SNDFILE *input, SNDFILE *output);

#endif

