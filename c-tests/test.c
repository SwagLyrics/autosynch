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

static int callback(const void *input, void *output, unsigned long framecount, 
        const PaStreamCallbackTimeInfo *timeinfo, PaStreamCallbackFlags statusflags, void *userdata) {
    
    float *out = (float *)output;
    callback_data_s *p_data = (callback_data_s *)userdata;
    sf_count_t num_read;

    memset(out, 0, sizeof(float) * framecount * p_data->info.channels);
    num_read = sf_read_float(p_data->file, out, framecount * p_data->info.channels);
    num_read /= 2;

    if (num_read < framecount)
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

    if (Pa_OpenDefaultStream(&stream, 0, data.info.channels, paFloat32, data.info.samplerate,
                FRAMES_PER_BUFFER, callback, &data) != paNoError)
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
