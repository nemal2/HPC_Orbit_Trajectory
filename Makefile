CC = gcc
CFLAGS = -O2
LIBS = -lm

all: serial

serial: rocket_serial.c rocket_physics.c rocket_utils.c
	$(CC) $(CFLAGS) rocket_serial.c rocket_physics.c rocket_utils.c -o serial $(LIBS)

clean:
	rm -f serial