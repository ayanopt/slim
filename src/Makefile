CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -march=native -ffast-math -pthread -flto
SRCDIR = train-utils
OBJDIR = obj

SOURCES = $(wildcard $(SRCDIR)/*.cpp)
OBJECTS = $(SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)

TARGET = llm-demo

all: $(TARGET)

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): llm-demo.cpp $(OBJECTS)
	$(CXX) $(CXXFLAGS) llm-demo.cpp $(OBJECTS) -o $(TARGET)

clean:
	rm -rf $(OBJDIR) $(TARGET)

debug: CXXFLAGS = -std=c++17 -Wall -Wextra -g -pthread
debug: clean all

.PHONY: all clean debug
