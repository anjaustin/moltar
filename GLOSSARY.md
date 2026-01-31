# Glossary

Technical terms and definitions used in the moltar research platform.

## A

### ADB (Android Debug Bridge)
A command-line tool that lets you communicate with an Android device. Used for installing apps, running shell commands, and debugging.

### API Level
An integer value that identifies the framework API revision offered by a version of the Android platform. Higher numbers indicate more recent releases.

### APK (Android Package Kit)
The file format used for distributing and installing applications on Android devices. Contains compiled code, resources, and manifest.

## B

### Battery Drain
The rate at which a device's battery discharges during operation. Measured as percentage per hour or milliamp-hours (mAh) per hour.

### Bootloader
A program that loads the operating system kernel during device startup. Can be locked (restricted) or unlocked (allows custom ROMs).

## C

### CI/CD (Continuous Integration/Continuous Deployment)
Automated processes for building, testing, and deploying software changes. Ensures code quality and rapid iteration.

### Code of Conduct
A set of rules outlining the norms, rules, and responsibilities of participants in a community or project.

### CPU (Central Processing Unit)
The primary processor in a computing device, responsible for executing instructions and performing calculations.

## D

### DSP (Digital Signal Processor)
A specialized microprocessor optimized for digital signal processing tasks, such as audio and video processing. In mobile SoCs like Snapdragon 480, DSPs (like Hexagon) provide hardware acceleration for AI inference.

### Dot Product Instructions
ARMv8.2-A CPU instructions (UDOT/SDOT) that perform fused multiply-add operations on vectors, providing significant performance improvements for quantized neural network computations.

### Device Tree
A data structure describing the hardware components of a computer system, used by the Linux kernel to identify and configure devices.

## E

### Embedded System
A computer system designed to perform dedicated functions within a larger mechanical or electrical system.

### ExecuTorch
A PyTorch runtime optimized for edge devices, providing efficient on-device inference for AI models.

### ExportPass
An ExecuTorch transformation class that modifies computation graphs during the export process, used for optimizations like quantization and hardware-specific transformations.

### Experimental Protocol
A detailed plan specifying the procedures, materials, and methods for conducting a scientific experiment.

## F

### Falsification
The act of disproving a hypothesis or theory by providing evidence that contradicts it. A core principle of scientific methodology used in the moltar research framework for validating performance claims.

### FX Graph
PyTorch's intermediate representation for neural networks, used by torch.fx for graph transformations and optimizations during model export.

### Fastboot
A protocol used for communicating with Android devices in bootloader mode, allowing flashing of partitions and system images.

### Framework API
The set of classes, interfaces, and methods provided by the Android framework for application development.

## G

### GPU (Graphics Processing Unit)
A specialized processor designed for rendering graphics and performing parallel computations.

### Ghost Partition Bug
A critical ExecuTorch bug where the XNNPack partitioner accepts operations for delegation but fails to actually move them to the DSP subgraph, preventing hardware acceleration.

### Gradle
A build automation tool used for Android development, managing dependencies, compilation, and packaging.

## H

### Hardware Acceleration
The use of specialized hardware components (like GPUs or DSPs) to perform computations faster than software running on general-purpose CPUs.

### Hexagon DSP
Qualcomm's DSP architecture used in Snapdragon SoCs for efficient AI inference and signal processing, providing hardware acceleration for neural network operations.

### Hypothesis
A testable statement or prediction about the relationship between variables in a research study.

## I

### IDE (Integrated Development Environment)
A software application that provides comprehensive facilities for software development, including editing, debugging, and build tools.

### Inference
The process of using a trained AI model to make predictions or decisions on new data.

### LFN (Liquid Foundation Model)
Liquid AI's family of foundation models optimized for conversational AI and real-time inference on edge devices, featuring continuous learning capabilities and temporal coherence.

### Instrumentation
The process of adding monitoring and logging capabilities to code for performance analysis and debugging.

## J

### JNI (Java Native Interface)
A framework that allows Java code to call native C/C++ libraries and vice versa, enabling performance-critical operations.

### LFN XNNPack Cleanup Pass
A custom ExecuTorch ExportPass that fixes XNNPack partitioning issues for Liquid Foundation Models, addressing tuple output problems and redundant quantization chains.

### JVM (Java Virtual Machine)
A virtual machine that executes Java bytecode, providing platform independence for Java applications.

## K

### Kernel
The core part of an operating system that manages system resources and provides low-level services to applications.

### Kotlin
A modern programming language that runs on the JVM and is officially supported for Android development.

## L

### Latency
The time delay between a request and its response. In AI contexts, the time from input to output generation.

### LFM (Liquid Foundation Model)
AI models developed by Liquid.ai, designed for efficient inference on edge devices with continuous learning capabilities.

### Lifecycle
The series of states an Android component (like an Activity) goes through from creation to destruction.

## M

### Manifest
An XML file in Android applications that defines essential information about the app, including permissions, components, and metadata.

### Memory Footprint
The amount of memory (RAM) used by an application or process during execution.

### Methodology
The systematic study of methods used in a particular field. In research contexts, the framework for conducting valid investigations.

## N

### Native Code
Code written in C/C++ that runs directly on the device's hardware, without interpretation by a virtual machine.

### Neural Network
A computing system inspired by biological neural networks, used for machine learning and AI applications.

### NDK (Native Development Kit)
A set of tools allowing developers to implement parts of their Android apps using native code languages like C and C++.

## O

### On-Device AI
Artificial intelligence processing that occurs directly on the device, without requiring cloud connectivity.

### Open Source
Software with source code available to the public for use, modification, and redistribution.

## P

### Performance Profiling
The process of measuring and analyzing the performance characteristics of software, including execution time and resource usage.

### Permissions
Access rights requested by Android applications to use device features or access sensitive data.

### Pre-Registration
The practice of documenting research plans and hypotheses before data collection to prevent bias and ensure transparency.

## Q

### Quantization
The process of reducing the precision of numerical values in AI models to decrease model size and improve inference speed.

### Quick Start
A streamlined setup process designed to get users up and running with minimal configuration.

## R

### Reproducibility
The ability to replicate research results using the same methods and data. A cornerstone of scientific research.

### Research Framework
A structured approach to conducting scientific investigations, including methodology, tools, and validation procedures.

### Root Access
Administrative privileges on Android devices, allowing full system access beyond standard user permissions.

## S

### Scientific Method
A systematic approach to investigation involving observation, hypothesis formation, experimentation, and conclusion.

### Snapdragon
A series of mobile processors designed by Qualcomm for smartphones and other devices.

### Snapdragon 480
Qualcomm's mobile SoC featuring 2x Cortex-A76 (big) cores and 6x Cortex-A55 (little) cores, with Hexagon 686 DSP for AI acceleration and Adreno 619 GPU.

### SpaceGhost
A research initiative optimizing ExecuTorch for Liquid AI Foundation Models on Motorola Snapdragon devices, implementing hardware-specific improvements and framework fixes.

### Statistical Significance
The likelihood that a result is not due to chance. Used to determine if research findings are meaningful.

## T

### Throughput
The rate at which a system processes requests or data. In AI contexts, often measured as inferences per second.

### Tokenization
The process of breaking text into smaller units (tokens) for natural language processing tasks.

## U

### Unit Testing
A software testing method where individual units or components of code are tested in isolation.

### USB Debugging
A developer option in Android that allows devices to communicate with development tools over USB.

## V

### Validation
The process of checking whether research results are accurate and reliable. Includes both internal and external validation methods.

### Virtual Environment
An isolated Python environment that allows packages to be installed for a specific project without affecting the system Python installation.

### Virtual Machine
Software that emulates a computer system, allowing applications to run in an isolated environment.

## W

### Wearables
Portable electronic devices worn on the body, such as smartwatches and fitness trackers.

### Workflow
A defined sequence of steps for completing a task or process, often automated in development environments.

## Z

### Zero-Day
A software vulnerability unknown to the vendor, often exploited by attackers before a fix is available.

---

*This glossary provides definitions for technical terms used throughout the moltar documentation. Terms are added as new concepts are introduced.*