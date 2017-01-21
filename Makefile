CPPFLAGS=-I/usr/local/cuda/include
CFLAGS=-std=gnu99 -g -Wall
LDLIBS=-lcudart -lcudnn
LDFLAGS=-L/usr/local/cuda/lib64

all: main

main: timer.o

run: main
	CUDA_VISIBLE_DEVICES=3 ./main

clean:
	rm -rf *.o main
