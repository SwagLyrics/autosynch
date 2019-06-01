#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <sndfile.h>
#include <portaudio.h>

#define M_PI (3.14159265358979323846)
#define FRAMES_PER_BUFFER (1024)

static int cutoff = 300;

typedef struct {
    SNDFILE *file;
    SF_INFO info;
} callback_data_s;

static void die(const char *s) { perror(s); exit(1); }

static int lpf(const void *input, void *output, unsigned long framecount,
        const PaStreamCallbackTimeInfo *timeinfo, PaStreamCallbackFlags statusflags,
        void *userdata) {

    sf_count_t count;
    float *p_output = (float *)output;
    callback_data_s *p_data = (callback_data_s *)userdata;

    float p_input[sizeof(float) * framecount * p_data->info.channels];
    count = sf_readf_float(p_data->file, p_input, framecount * p_data->info.channels);

    memset(p_output, 0, sizeof(float) * framecount * p_data->info.channels);

    float RC = 1.0 / (cutoff*2*M_PI);
    float dt = 1.0 / p_data->info.samplerate;
    float alpha = dt / (RC+dt);

    p_output[0] = p_input[0];
    for (int i = 1; i < framecount; i++)
        p_output[i] = p_output[i-1] + alpha * (p_input[i]-p_output[i-1]);

    if (count < framecount)
        return paComplete;
    return paContinue;
}

static int callback(const void *input, void *output, unsigned long framecount,
        const PaStreamCallbackTimeInfo *timeinfo, PaStreamCallbackFlags statusflags,
        void *userdata) {

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

    printf("%d\n", data.info.channels);

    if (Pa_Initialize() != paNoError)
        die("Pa_Initialize() failed");

    if (Pa_OpenDefaultStream(&stream, 0, data.info.channels, paFloat32,
                data.info.samplerate, paFramesPerBufferUnspecified, lpf, &data) != paNoError)
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
