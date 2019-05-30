#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sndfile.h>
#include <portaudio.h>

#include "mix-to-mono.h"

#define M_PI (3.14159265358979323846)
#define FRAMES_PER_BUFFER (1024)

typedef struct {
    SNDFILE *file;
    SF_INFO info;
} callback_data_s;

static void die(const char *s) { perror(s); exit(1); }

static void deinterleave(float *input, float **output, int framecount, int channels) {
    for (int i = 0; i < channels; i++)
        for (int j = 0; j < framecount; j++) 
            output[i][j] = input[j*channels+i];
}

static int callback(const void *input, void *output, unsigned long framecount, 
        const PaStreamCallbackTimeInfo *timeinfo, PaStreamCallbackFlags statusflags,
        void *userdata) {

    sf_count_t frames_read;
    callback_data_s *p_data = (callback_data_s *)userdata;

    float *p_output = (float *)output;
    memset(p_output, 0, sizeof(float) * framecount);

    float p_input[sizeof(float) * framecount * p_data->info.channels];
    float mono[framecount];
    float split[p_data->info.channels][framecount];

    frames_read = sf_readf_float(p_data->file, p_input, framecount);
    
    float mix;
    for (int i = 0; i < frames_read; i++) {
        mix = 0.0;
        for (int j = 0; j < p_data->info.channels; j++) {
            mix += p_input[i * p_data->info.channels + j];
            split[j][i] = p_input[i * p_data->info.channels + j];
        }
        mono[i] = mix / p_data->info.channels;
    





    sf_count_t count;
    float *p_output = (float *)output;
    callback_data_s *p_data = (callback_data_s *)userdata;

    float p_input[sizeof(float) * framecount * p_data->info.channels];
    count = sf_readf_float(p_data->file, p_input, framecount * p_data->info.channels); 

    memset(p_output, 0, sizeof(float) * framecount * p_data->info.channels);

}
 
int main(int argc, char **argv) {
    
    if (argc != 2) {
        printf("usage: %s <input_file.wav>\n", argv[0]);
        exit(1);
    }

    char *fname = argv[1];

    callback_data_s data;
    PaStream *stream;

    memset(&data.info, 0, sizeof(data.info));
    if ((data.file = sf_open(fname, SFM_READ, &data.info)) == NULL)
        die("sf_open() failed");

    if (data.info.channels != 2) {
        fprintf(stderr, "File must have exactly 2 channels\n");
        exit(1);
    }

    if (Pa_Initialize() != paNoError)
        die("Pa_Initialize() failed");

    if (Pa_OpenDefaultStream(&stream, 0, data.info.channels, paFloat32, data.info.samplerate,
                paFramesPerBufferUnspecified, callback, &data) != paNoError)
        die("Pa_OpenDefaultStream() failed");

    if (Pa_StartStream(stream) != paNoError)
        die("Pa_StartStream() failed");

    while (Pa_IsStreamActive(stream))
        Pa_Sleep(100);

    if (Pa_CloseStream(stream) != paNoError)
        die("Pa_CloseStream() failed");
    if (Pa_Terminate() != paNoError)
        die("Pa_Terminate() failed");

    sf_close(data.file);
}
