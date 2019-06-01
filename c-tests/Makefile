CC = clang

INCLUDES = -I/usr/local/include

CFLAGS = -g -Wall $(INCLUDES)

LDFLAGS = -g -L/usr/local/lib -L/usr/local/Cellar
LDLIBS = -lsndfile -lportaudio

<<<<<<< HEAD
isolate-center: isolate-center.o mix-to-mono.o

isolate-center.o: mix-to-mono.h

mix-to-mono.o: mix-to-mono.h

.PHONY: clean
clean:
	rm -f *~ a.out core *.o isolate-center 
=======
test: test.o

test.o:

.PHONY: clean
clean:
	rm -f *~ a.out core *.o test
>>>>>>> 17c58b609462c51af18dc5cf9e9234b986844256

.PHONY: all
all: clean test
