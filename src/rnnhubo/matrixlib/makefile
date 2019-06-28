

LIB=libcblas.a libatlas.a

all:main

main:main.c
	gcc -o $@ $^ $(LIB)  -lm -pthread -ldl


.PHONY:

clean:
	rm -f main
