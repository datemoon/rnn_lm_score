
GXX=g++

#CXXFLAGS=-g -msse -msse2 -Wall -pthread -rdynamic -std=c++98
CXXFLAGS=-msse -msse2 -Wall -I.. -pthread -DKALDI_DOUBLEPRECISION=0 -DHAVE_POSIX_MEMALIGN -Wno-sign-compare -Wno-unused-local-typedefs -Winit-self -DHAVE_EXECINFO_H=1 -rdynamic -DHAVE_CXXABI_H -std=c++0x 
CXXFLAGS += -fPIC -ffast-math -funroll-loops -g
CXXFLAGS += -DCBLAS -DRNNLMRESOCRE #-DDEBUG 
LDLIBS= rnnhubo/matrixlib/libcblas.a rnnhubo/matrixlib/libatlas.a \
		-lm -lpthread -ldl
INC=-I . -I ./ngram/ -I ./rnnhubo/ -I rnnhubo/matrixlib/ -I pthread

SOURCE=front_tree.cpp ngram/FsmLM.cpp rnnhubo/rnn.cpp

OBJ=front_tree.o

all:main main_list main_list_pthread librnnlm.a 


main:main.cpp $(SOURCE)
	$(GXX) $(INC) $(CXXFLAGS) -o $@ $^  $(LDLIBS)

main_list:main_list.cpp $(SOURCE)
	$(GXX) $(INC) $(CXXFLAGS) -o $@ $^  $(LDLIBS)

main_list_pthread:pthread/main_list_pthread.cpp $(SOURCE)
	$(GXX) $(INC) $(CXXFLAGS) -o $@ $^ pthread/pthread_hb.c  $(LDLIBS)

librnnlm.a:front_tree.o rnnhubo/rnn.o front_tree_api.o
	ar -rcu  $@ $^ 

%.o:%.cpp
	$(GXX) $(INC) -I ../HawkDecoder/libDecode/include $(CXXFLAGS) -c -o $@ $^

.PHONY:

clean:
	rm -f main main_list main_list_pthread front_tree.o rnnhubo/rnn.o front_tree_api.o librnnlm.a
