CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -pthread
SRCDIR = train-utils
OBJDIR = obj

# Source files
SOURCES = $(wildcard $(SRCDIR)/*.cpp)
OBJECTS = $(SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)

# Target executables
TRAIN_TARGET = train-model
LLM_TARGET = llm-demo

# Default target
all: $(TRAIN_TARGET) $(LLM_TARGET)

# Create object directory
$(OBJDIR):
	mkdir -p $(OBJDIR)

# Compile object files
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Link train-model executable
$(TRAIN_TARGET): train-model.cpp $(OBJECTS)
	$(CXX) $(CXXFLAGS) train-model.cpp $(OBJECTS) -o $(TRAIN_TARGET)

# Link llm-demo executable
$(LLM_TARGET): llm-demo.cpp $(OBJECTS)
	$(CXX) $(CXXFLAGS) llm-demo.cpp $(OBJECTS) -o $(LLM_TARGET)

# Clean build files
clean:
	rm -rf $(OBJDIR) $(TRAIN_TARGET) $(LLM_TARGET)

.PHONY: all clean