CC = clang

INCLUDES = -I/usr/local/include

CFLAGS = -g -Wall $(INCLUDES)

LDFLAGS = -g -L/usr/local/lib -L/usr/local/Cellar
LDLIBS = -lsndfile -lportaudio

isolate-center: isolate-center.o mix-to-mono.o

isolate-center.o: mix-to-mono.h

mix-to-mono.o: mix-to-mono.h

.PHONY: clean
clean:
	rm -f *~ a.out core *.o isolate-center 

.PHONY: all
all: clean test
