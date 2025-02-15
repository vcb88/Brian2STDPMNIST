# Testing Guide

This document describes the testing procedures and available commands for evaluating the network's performance.

## Available Test Commands

### 1. Full Testing
Tests the network on the complete test dataset (10,000 examples)
```bash
make container-test
```

### 2. Quick Testing
Tests on a fixed subset of 1000 examples
```bash
make container-test-quick
```

### 3. Random Testing
Tests on a random subset of 1000 examples
```bash
make container-test-random
```

### 4. Custom Size Testing
Tests on a specific number of examples
```bash
make container-test-size SIZE=500
```

### 5. Random Size Testing
Tests on a random subset of specific size
```bash
make container-test-size-random SIZE=500
```

### 6. Custom Testing
Allows full customization of test parameters
```bash
make container-test-custom TEST_ARGS="--test-size 500 --random-subset --verbose"
```

## Test Parameters

### Required Parameters
- `SIZE`: Number of examples to test (for size-specific commands)

### Optional Parameters
- `--test-size N`: Number of examples to test
- `--random-subset`: Use random subset of examples
- `--verbose`: Enable verbose output
- `--data-dir PATH`: Custom data directory path

## Running Tests

### 1. Basic Testing
```bash
# Full test
make container-test

# Quick test
make container-test-quick
```

### 2. Custom Size Testing
```bash
# Test on 500 examples
make container-test-size SIZE=500

# Test on 1000 random examples
make container-test-size-random SIZE=1000
```

### 3. Advanced Testing
```bash
# Custom test with specific parameters
make container-test-custom TEST_ARGS="--test-size 500 --random-subset --verbose"
```

## Test Output

### 1. Console Output
- Progress indicators
- Error messages
- Performance metrics

### 2. Log Files
- Detailed test logs
- Error logs
- Performance data

### 3. Results
- Classification accuracy
- Confusion matrix
- Per-digit performance

## Test Environment

### 1. Docker Container
```bash
# Start container
make docker-run

# Stop container
make docker-stop
```

### 2. Dataset
```bash
# Download dataset
make dataset-download

# Prepare dataset
make dataset-prepare

# Check status
make dataset-status
```

## Troubleshooting

### 1. Common Issues
- Missing dataset
- Weight file errors
- Memory issues

### 2. Solutions
- Check dataset presence
- Verify weight files
- Reduce test size

### 3. Debug Mode
```bash
make container-test-custom TEST_ARGS="--test-size 100 --verbose"
```

## Performance Analysis

### 1. Quick Analysis
Basic performance metrics are printed after testing

### 2. Detailed Analysis
Run analysis script for detailed metrics:
```bash
python analyze_results.py
```

### 3. Visualization
Generate performance visualizations:
```bash
python visualize_results.py
```

## Best Practices

### 1. Testing Workflow
1. Start with quick tests
2. Validate on random subsets
3. Run full tests if needed

### 2. Resource Management
- Use appropriate test size
- Monitor memory usage
- Clean up after testing

### 3. Results Validation
- Check error rates
- Compare with baselines
- Verify sample diversity

## Configuration

### 1. Test Parameters
Located in `config/test_config.py`

### 2. Network Parameters
Located in `config/network_config.py`

### 3. Docker Settings
Located in `docker-compose.yml`

## Additional Commands

### 1. Environment Reset
```bash
make reset-and-test
```

### 2. Clean Environment
```bash
make clean
```

### 3. Update Code
```bash
make update
```