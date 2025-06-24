CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -O2

SRC := src/main.cpp
OUTDIR := build
TARGET := $(OUTDIR)/main

.PHONY: all

all: $(TARGET)

$(TARGET): $(SRC)
	@mkdir -p $(OUTDIR)
	$(CXX) $(CXXFLAGS) $< -o $@

.PHONY: run clean

run: all
	@$(TARGET)

clean:
	rm -rf $(OUTDIR)
