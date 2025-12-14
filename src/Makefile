CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -march=native -ffast-math -pthread -flto -funroll-loops
SRCDIR = train-utils
OBJDIR = obj

SOURCES = $(wildcard $(SRCDIR)/*.cpp)
OBJECTS = $(SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)

TARGET = slim

all: $(TARGET)

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): llm-demo.cpp $(OBJECTS)
	$(CXX) $(CXXFLAGS) llm-demo.cpp $(OBJECTS) -o $(TARGET)

clean:
	rm -rf $(OBJDIR) $(TARGET)

debug: CXXFLAGS = -std=c++17 -Wall -Wextra -g -pthread -fsanitize=address
debug: clean all

.PHONY: all clean debug
