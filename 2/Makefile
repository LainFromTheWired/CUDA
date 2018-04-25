NVCC=nvcc 
CC=g++   

CUDA_TARGET=-arch=sm_20
CUDA_CONFIG=`OcelotConfig -l`
CUDA_FLAGS=-c -g 
CC_FLAGS=-o

SRC=$(wildcard *.cu)
OBJ=$(patsubst %.cu, %.o, $(SRC))

OUT=classsificator

all:  $(OUT)

$(OUT): $(OBJ)
	$(CC)  $(CC_FLAGS)  $(OUT) $(OBJ) $(CUDA_CONFIG)

$(OBJ) : %.o : %.cu 
	$(NVCC)  $(CUDA_FLAGS) $< $(CUDA_TARGET) 

clean: 
	rm -rf $(OBJ) 