nvcc -c main.cu -arch=sm_20
g++ -o main.out main.o  `OcelotConfig -l`

./main.out