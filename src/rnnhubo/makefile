

CXXFLAGS=-g -I. -Imatrixlib -DCBLAS 

GXX=g++

SOURCE=rnn.cpp
LDLIBS=matrixlib/libcblas.a matrixlib/libatlas.a -lm -lpthread -ldl

all:main

main:main.cpp $(SOURCE)
	$(GXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS) 


.PHONY:

clean:
	rm -f main

