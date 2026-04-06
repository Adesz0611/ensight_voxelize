CC = gcc
CFLAGS = -O3 -march=native -fopenmp -Wall -Wextra
INCLUDES = -Ilibs/cfd_lib
LDFLAGS = -lm

PROGNAME = voxelize

SRCS = main.c \
	   lz4.c

all: $(PROGNAME)

debug: CFLAGS = -Wall -Wextra -g -Og -march=native -fopenmp -DDEBUG
debug: all

$(PROGNAME): $(SRCS)
	$(CC) $(CFLAGS) $(INCLUDES) $(SRCS) -o $(PROGNAME) $(LDFLAGS)

clean:
	rm -f $(PROGNAME)

.PHONY: all clean debug
