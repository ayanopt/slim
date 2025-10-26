CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -pthread
SRCDIR = train-utils
OBJDIR = obj

# Source files
SOURCES = $(wildcard $(SRCDIR)/*.cpp)
OBJECTS = $(SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)

LLM_TARGET = llm-demo

# Default target
all: $(TRAIN_TARGET) $(LLM_TARGET)

# Create object directory
$(OBJDIR):
	mkdir -p $(OBJDIR)

# Compile object files
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@



# Link llm-demo executable
$(LLM_TARGET): llm-demo.cpp $(OBJECTS)
	$(CXX) $(CXXFLAGS) llm-demo.cpp $(OBJECTS) -o $(LLM_TARGET)

# Clean build files
clean:
	rm -rf $(OBJDIR) $(LLM_TARGET)

.PHONY: all clean