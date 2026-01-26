# Development Guide

This guide provides comprehensive information for developers contributing to the moltar research platform.

## Development Environment Setup

### Prerequisites

#### System Requirements
- **macOS**: 11.0+ (12.0+ recommended)
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 50GB free space for development
- **Network**: Stable internet for dependencies

#### Required Software
```bash
# Core development tools
brew install git python@3.11 openjdk@17 android-studio

# Android development
brew install android-platform-tools

# Python development
pip install virtualenv black flake8 mypy pytest

# Additional tools
brew install jq yq gh hub
```

### Environment Configuration

#### Shell Configuration
Add to `~/.zshrc` or `~/.bashrc`:
```bash
# Moltar development environment
export MOLTA_ROOT="$HOME/Projects/moltar"
export PYTHONPATH="$MOLTA_ROOT:$PYTHONPATH"

# Android development
export ANDROID_HOME="$HOME/Library/Android/sdk"
export ANDROID_NDK_HOME="$ANDROID_HOME/ndk/$(ls $ANDROID_HOME/ndk | head -1)"
export PATH="$PATH:$ANDROID_HOME/tools:$ANDROID_HOME/platform-tools"

# Java for Android development
export JAVA_HOME="/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"

# Python virtual environment
export VIRTUAL_ENV="$MOLTA_ROOT/venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
```

#### Git Configuration
```bash
# Set up Git with your credentials
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Enable useful Git features
git config --global init.defaultBranch main
git config --global core.editor "code --wait"
git config --global pull.rebase true
```

### Project Setup

#### Clone and Initialize
```bash
# Clone the repository
git clone https://github.com/anjaustin/moltar.git
cd moltar

# Initialize development environment
./moltar_setup.sh

# Or manual setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Verify Setup
```bash
# Check environment
python3 --version
java -version
adb version
git --version

# Run basic tests
./moltar --help
python3 -c "import torch; print('PyTorch version:', torch.__version__)"
```

## Development Workflow

### Branching Strategy

#### Branch Naming Convention
```
feature/<description>     # New features
bugfix/<description>      # Bug fixes
hotfix/<description>      # Critical fixes
research/<description>    # Research implementations
docs/<description>        # Documentation updates
```

#### Example Workflow
```bash
# Create feature branch
git checkout -b feature/add-performance-monitoring

# Make changes
# ... development work ...

# Commit changes
git add .
git commit -m "feat: add performance monitoring dashboard

- Add real-time latency tracking
- Implement memory usage graphs
- Include battery drain monitoring
- Support custom metric collection

Closes #123"

# Push and create PR
git push -u origin feature/add-performance-monitoring
```

### Code Standards

#### Python Code Style
```bash
# Formatting with Black
black .

# Linting with Flake8
flake8 . --max-line-length=100 --extend-ignore=E203,W503

# Type checking with MyPy
mypy . --ignore-missing-imports
```

#### Kotlin Code Style
- Follow Android Kotlin style guide
- Use ktlint for formatting
- Enable Android Studio's built-in linting

#### Shell Script Standards
```bash
# Use shellcheck for validation
brew install shellcheck
shellcheck scripts/*.sh

# Script template
#!/bin/bash
set -euo pipefail  # Strict error handling

# Function definitions
main() {
    # Main logic
}

main "$@"
```

### Testing

#### Unit Tests
```bash
# Run Python tests
pytest tests/ -v --cov=src --cov-report=html

# Run Android tests
./gradlew testDebugUnitTest

# Run integration tests
pytest tests/integration/ -v
```

#### Performance Testing
```bash
# Run performance benchmarks
python3 scripts/benchmark_performance.py

# Profile memory usage
python3 -m memory_profiler scripts/profile_memory.py

# Test with real device
./research/brack/scripts/test_brack_deployment.sh
```

#### Falsification Testing
```bash
# Run scientific validation
./research/brack/scripts/falsify_performance_claims.sh

# Generate test reports
./scripts/generate_test_report.sh
```

## Component Development

### Adding a New Research Module

#### 1. Create Module Structure
```bash
# Create module directory
mkdir -p research/new_module/{src,tests,docs}

# Create basic files
touch research/new_module/{README.md,__init__.py}
touch research/new_module/src/module.py
touch research/new_module/tests/test_module.py
```

#### 2. Implement Core Functionality
```python
# research/new_module/src/module.py
"""
New research module for [specific functionality].
"""

class NewResearchModule:
    """Main research module class."""

    def __init__(self, config: dict):
        """Initialize the research module."""
        self.config = config
        self.validate_config()

    def validate_config(self):
        """Validate configuration parameters."""
        required_keys = ['input_path', 'output_path']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

    def run_experiment(self) -> dict:
        """Run the research experiment."""
        # Implementation
        return {
            'status': 'completed',
            'results': {},
            'metrics': {}
        }
```

#### 3. Add Tests
```python
# research/new_module/tests/test_module.py
import pytest
from research.new_module.src.module import NewResearchModule

class TestNewResearchModule:
    def test_initialization_valid_config(self):
        config = {
            'input_path': '/tmp/input',
            'output_path': '/tmp/output'
        }
        module = NewResearchModule(config)
        assert module.config == config

    def test_initialization_missing_config(self):
        config = {}
        with pytest.raises(ValueError, match="Missing required config key"):
            NewResearchModule(config)

    def test_run_experiment(self):
        config = {
            'input_path': '/tmp/input',
            'output_path': '/tmp/output'
        }
        module = NewResearchModule(config)
        result = module.run_experiment()

        assert result['status'] == 'completed'
        assert 'results' in result
        assert 'metrics' in result
```

#### 4. Add Documentation
```markdown
# New Research Module

## Overview

Brief description of the research module and its purpose.

## Configuration

```json
{
  "input_path": "/path/to/input",
  "output_path": "/path/to/output",
  "optional_param": "default_value"
}
```

## Usage

```python
from research.new_module.src.module import NewResearchModule

config = {
    'input_path': '/data/input',
    'output_path': '/data/output'
}

module = NewResearchModule(config)
results = module.run_experiment()
```

## API Reference

### NewResearchModule

#### Methods

- `__init__(config: dict)`: Initialize the module
- `run_experiment() -> dict`: Execute the research experiment
```

#### 5. Register Module
```python
# research/__init__.py or main registry
from .new_module.src.module import NewResearchModule

__all__ = ['NewResearchModule']
```

### Android Development

#### Setting Up Android Project
```bash
# Create new Android module
cd research/brack/src
./gradlew init --type basic-library --dsl kotlin --package com.moltar.newmodule

# Add to settings.gradle.kts
include(":newmodule")
```

#### Android Testing
```bash
# Run unit tests
./gradlew :newmodule:testDebugUnitTest

# Run instrumentation tests
./gradlew :newmodule:connectedDebugAndroidTest

# Generate test coverage
./gradlew :newmodule:jacocoTestReport
```

### Documentation Development

#### Adding New Documentation
```bash
# Create documentation file
touch docs/new_feature.md

# Add to table of contents
echo "- [New Feature](docs/new_feature.md)" >> docs/README.md
```

#### Documentation Standards
- Use Markdown format
- Include code examples
- Provide troubleshooting sections
- Keep documentation current with code changes

## Quality Assurance

### Code Review Checklist
- [ ] Code follows style guidelines
- [ ] Unit tests added and passing
- [ ] Documentation updated
- [ ] Breaking changes documented
- [ ] Performance impact assessed
- [ ] Security implications reviewed

### Pre-Commit Checks
```bash
# Run all quality checks
./scripts/pre_commit_checks.sh

# Or run individually
black --check .
flake8 .
mypy .
pytest tests/
shellcheck scripts/*.sh
```

### CI/CD Integration

#### GitHub Actions Workflow
```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]

jobs:
  test:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest tests/ -v
      - name: Check code quality
        run: |
          black --check .
          flake8 .
```

## Debugging and Troubleshooting

### Common Development Issues

#### Import Errors
```python
# Check Python path
python3 -c "import sys; print('\n'.join(sys.path))"

# Activate virtual environment
source venv/bin/activate

# Reinstall package
pip uninstall package_name
pip install package_name
```

#### Android Build Issues
```bash
# Clean build
./gradlew clean

# Clear caches
rm -rf ~/.gradle/caches/
./gradlew cleanBuildCache

# Check Java version
java -version
echo $JAVA_HOME
```

#### Device Connection Issues
```bash
# Restart ADB
adb kill-server
adb start-server

# Check device
adb devices -l

# Verify permissions
ls -la /usr/local/bin/adb
```

### Performance Debugging

#### Memory Profiling
```python
# Add memory profiling
from memory_profiler import profile

@profile
def memory_intensive_function():
    # Code to profile
    pass
```

#### Performance Benchmarking
```python
import time

def benchmark_function(func, *args, **kwargs):
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Function executed in {execution_time:.4f} seconds")
    return result
```

## Release Process

### Version Management
```bash
# Update version
echo "1.1.0" > VERSION

# Update changelog
vim CHANGELOG.md

# Create release commit
git add VERSION CHANGELOG.md
git commit -m "chore: release version 1.1.0"
git tag v1.1.0
```

### Release Checklist
- [ ] All tests passing
- [ ] Documentation updated
- [ ] Changelog complete
- [ ] Version numbers updated
- [ ] Release notes written
- [ ] Breaking changes documented

### Publishing Release
```bash
# Push release
git push origin main
git push origin v1.1.0

# Create GitHub release
gh release create v1.1.0 --title "Release v1.1.0" --notes-file RELEASE_NOTES.md
```

## Contributing

### Getting Help
- **Issues**: Report bugs and request features
- **Discussions**: Ask questions and share ideas
- **Documentation**: Check existing docs first
- **Code Reviews**: Request feedback on PRs

### Recognition
Contributors are recognized in:
- Git commit history
- CHANGELOG.md for significant changes
- Repository contributors list
- Release notes

---

*This development guide ensures consistent, high-quality contributions to the moltar research platform. Follow these guidelines to maintain code quality and research integrity.*