#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sndfile.h>
#include <portaudio.h>

#define FRAMES_PER_BUFFER (1024)

typedef struct {
    SNDFILE *file;
    SF_INFO info;
} callback_data_s;

static void die(const char *s) { perror(s); exit(1); }

//Need to fix: p_input & segfault
sf_count_t lpf(void *output, unsigned long framecount,
        callback_data_s *p_data, int cutoff) {
    
    sf_count_t nread;
    float *p_input = malloc(sizeof(float) * framecount * p_data->info.channels); 
    float *p_output = (float *)output;

    double tau = 1.0 / cutoff;
    double alpha = framecount / tau;

    nread = sf_read_float(p_data->file, p_input, framecount * p_data->info.channels);
     
    float tmp = p_input[0];
    for (int i = 0; i < sizeof(float) * framecount * p_data->info.channels; i++) {
        tmp += alpha * (p_input[i]-tmp);
        p_output[i] = tmp;
    }

    free(p_input);
    return nread/2;
}

int callback(const void *input, void *output, unsigned long framecount, 
        const PaStreamCallbackTimeInfo *timeinfo, PaStreamCallbackFlags statusflags,
        void *userdata) {
    
    sf_count_t nread;
    float *p_output = (float *)output;
    callback_data_s *p_data = (callback_data_s *)userdata;

    memset(p_output, 0, sizeof(float) * framecount * p_data->info.channels);
    nread = lpf(output, framecount, p_data, 500);
    //nread = sf_read_float(p_data->file, p_output, framecount * p_data->info.channels);

    if (nread < framecount)
        return paComplete;
    return paContinue;
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

    if (Pa_Initialize() != paNoError)
        die("Pa_Initialize() failed");

    if (Pa_OpenDefaultStream(&stream, 0, data.info.channels, paFloat32,
                data.info.samplerate, FRAMES_PER_BUFFER, callback, &data) != paNoError)
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
