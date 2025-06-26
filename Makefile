CXX := g++
CXXFLAGS := -std=c++23 -Wall -Wextra -pedantic-errors -Werror -O2 -Isrc

SRC := $(shell find src -name '*.cpp')
OUTDIR := build
TARGET := $(OUTDIR)/main

.PHONY: all

all: $(TARGET)

$(TARGET): $(SRC)
	@mkdir -p $(OUTDIR)
	$(CXX) $(CXXFLAGS) $(SRC) -o $@

.PHONY: run clean

run: all
	@$(TARGET)

clean:
	rm -rf $(OUTDIR)

.PHONY: data

data:
	bash scripts/download_mnist.sh
