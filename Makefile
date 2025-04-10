NVCC        = nvcc
NVCC_FLAGS  = -O3 -std=c++20 -lcurand

TARGET      = sgd
SRC         = sgd.cu
OBJ         = $(SRC:.cu=.o)

all: $(TARGET)

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(PROFILE) -c $< -o $@

$(TARGET): $(OBJ)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

clean:
	rm -f $(TARGET) *.o
