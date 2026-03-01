NVCC      := nvcc
CXXFLAGS  := -O3
TARGET    := simplex.out

SOURCES := \
	simplex.cu \
	app/app.cu \
	parser/parser.cu \
	solver/kernels.cu \
	solver/solver.cu \
	io/io.cu

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(NVCC) $(CXXFLAGS) $(SOURCES) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
