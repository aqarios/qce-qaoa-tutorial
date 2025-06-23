# Get all .ipynb files from solutions directory
SOLUTIONS := $(wildcard solutions/*.ipynb)
# Generate corresponding tutorial filenames
TUTORIALS := $(patsubst solutions/%.ipynb,tutorials/%.ipynb,$(SOLUTIONS))

# Default target
all: $(TUTORIALS)

# Create tutorials directory if it doesn't exist
tutorials:
	mkdir -p tutorials

# Rule to process each notebook
tutorials/%.ipynb: solutions/%.ipynb sub.sh | tutorials
	scripts/sub.sh $< > $@

# Clean target to remove tutorials directory
clean:
	rm -rf tutorials

# Phony targets
.PHONY: all clean

# Print available targets
help:
	@echo "Available targets:"
	@echo "  all     - Process all notebooks from solutions/ to tutorials/"
	@echo "  clean   - Remove tutorials/ directory"
	@echo "  help    - Show this help message"
