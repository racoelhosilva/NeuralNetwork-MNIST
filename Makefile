CXX := g++
CXXFLAGS := -std=c++23 -Wall -Wextra -pedantic-errors -Werror -O2 -Isrc
CXXFLAGS += -MMD -MP

SRC := $(shell find src -name '*.cpp')
OBJ := $(patsubst src/%.cpp, build/%.o, $(SRC))
OUTDIR := build
TARGET := $(OUTDIR)/main

.PHONY: all run clean data

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@

build/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

-include $(OBJ:.o=.d)

run: all
	@$(TARGET)

clean:
	rm -rf $(OUTDIR)

data:
	@bash scripts/download_mnist.sh
