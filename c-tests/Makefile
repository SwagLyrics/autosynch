CC = clang

INCLUDES = -I/usr/local/include

CFLAGS = -g -Wall $(INCLUDES)

LDFLAGS = -g -L/usr/local/lib -L/usr/local/Cellar
LDLIBS = -lsndfile -lportaudio

test: test.o

test.o:

.PHONY: clean
clean:
	rm -f *~ a.out core *.o test

.PHONY: all
all: clean test
