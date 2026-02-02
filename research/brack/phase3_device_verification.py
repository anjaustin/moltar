#!/usr/bin/env python3
"""
Phase 3 Device Verification: Deploy and Test Integrated LFM Pipeline

Falsification methodology: Deploy Phase 3 integrated pipeline to MediaTek device
and rigorously test performance claims against hardware reality.

Tests:
- End-to-end inference latency (<300ms target)
- Peak memory usage (<280MB target)
- Accuracy preservation (>99% target)
- Hardware utilization (Mali GPU, ION memory)
- Thermal and power constraints

Falsification: Identify gaps between implementation and hardware reality.
"""

import time
import subprocess
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

class Phase3DeviceVerifier:
    """
    Rigorous device verification for Phase 3 integrated pipeline

    Deploys and tests the complete LFM2-350M pipeline on MediaTek + Mali hardware
    with falsification methodology to identify implementation vs. reality gaps.
    """

    def __init__(self, device_ip: str = "192.168.1.100", adb_port: int = 5555):
        self.device_ip = device_ip
        self.adb_port = adb_port
        self.device_connected = False
        self.verification_results = {}

        # Test configurations
        self.test_configs = {
            'latency_test': {
                'sequences': [64, 128, 256, 512],
                'runs_per_config': 5,
                'target_latency_ms': 300
            },
            'memory_test': {
                'target_memory_mb': 280,
                'memory_pressure_test': True
            },
            'accuracy_test': {
                'baseline_comparison': True,
                'accuracy_threshold': 0.99
            },
            'hardware_test': {
                'gpu_utilization_target': 90,
                'thermal_limit_c': 45,
                'power_budget_mw': 800
            }
        }

    def connect_to_device(self) -> bool:
        """
        Establish connection to MediaTek device

        Returns:
            True if connection successful
        """
        print("🔌 Connecting to MediaTek device...")

        try:
            # Connect via ADB
            cmd = f"adb connect {self.device_ip}:{self.adb_port}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                # Verify connection
                verify_cmd = "adb devices"
                verify_result = subprocess.run(verify_cmd, shell=True, capture_output=True, text=True)

                if self.device_ip in verify_result.stdout:
                    self.device_connected = True
                    print(f"✅ Connected to device at {self.device_ip}:{self.adb_port}")

                    # Get device info
                    self._get_device_info()
                    return True

            print(f"❌ Failed to connect: {result.stderr}")
            return False

        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False

    def _get_device_info(self):
        """Get device hardware information"""
        print("📱 Getting device information...")

        try:
            # Get device model
            model_cmd = "adb shell getprop ro.product.model"
            model_result = subprocess.run(model_cmd, shell=True, capture_output=True, text=True)
            device_model = model_result.stdout.strip()

            # Get Android version
            android_cmd = "adb shell getprop ro.build.version.release"
            android_result = subprocess.run(android_cmd, shell=True, capture_output=True, text=True)
            android_version = android_result.stdout.strip()

            # Get chipset info (MediaTek specific)
            chipset_cmd = "adb shell getprop ro.mediatek.platform"
            chipset_result = subprocess.run(chipset_cmd, shell=True, capture_output=True, text=True)
            chipset = chipset_result.stdout.strip()

            # Get memory info
            mem_cmd = "adb shell cat /proc/meminfo | grep MemTotal"
            mem_result = subprocess.run(mem_cmd, shell=True, capture_output=True, text=True)
            memory_info = mem_result.stdout.strip()

            device_info = {
                'model': device_model,
                'android_version': android_version,
                'chipset': chipset,
                'memory': memory_info,
                'expected_chipset': 'MT6855V',
                'expected_gpu': 'Mali-G52'
            }

            print(f"   Device: {device_info['model']}")
            print(f"   Android: {device_info['android_version']}")
            print(f"   Chipset: {device_info['chipset']}")
            print(f"   Memory: {device_info['memory']}")

            # Verify this is our target hardware
            if 'MT6855V' not in chipset:
                print("⚠️  Warning: Not the expected MediaTek MT6855V chipset")
                print("   This may affect performance validation accuracy")

            self.verification_results['device_info'] = device_info

        except Exception as e:
            print(f"❌ Failed to get device info: {e}")

    def deploy_pipeline(self) -> bool:
        """
        Deploy the integrated LFM pipeline to device

        Returns:
            True if deployment successful
        """
        print("📦 Deploying Phase 3 integrated pipeline...")

        if not self.device_connected:
            print("❌ Device not connected")
            return False

        try:
            # Build the executorch runner with integrated pipeline
            build_success = self._build_integrated_runner()
            if not build_success:
                print("❌ Build failed")
                return False

            # Push to device
            push_success = self._push_to_device()
            if not push_success:
                print("❌ Push to device failed")
                return False

            # Set permissions and verify
            setup_success = self._setup_device_environment()
            if not setup_success:
                print("❌ Device setup failed")
                return False

            print("✅ Pipeline deployment complete")
            return True

        except Exception as e:
            print(f"❌ Deployment failed: {e}")
            return False

    def _build_integrated_runner(self) -> bool:
        """Build the integrated LFM pipeline runner"""
        print("🔨 Building integrated LFM runner...")

        # Build script path
        build_script = Path("research/brack/build_integrated_lfm_runner.sh")

        if not build_script.exists():
            print(f"❌ Build script not found: {build_script}")
            return False

        try:
            # Make script executable
            os.chmod(build_script, 0o755)

            # Run build
            result = subprocess.run(str(build_script), cwd=Path("research/brack"),
                                  capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                print("✅ Build successful")
                return True
            else:
                print(f"❌ Build failed: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ Build error: {e}")
            return False

    def _push_to_device(self) -> bool:
        """Push built runner and models to device"""
        print("📤 Pushing to device...")

        files_to_push = [
            "research/brack/build/executorch_runner_integrated",
            "research/brack/models/lfm2_350m_quantized.pte",
            "research/brack/shaders/quantized_matmul.comp.spv",
            "research/brack/shaders/quantized_attention.comp.spv"
        ]

        device_path = "/data/local/tmp/neural_interposer_phase3"

        try:
            # Create device directory
            mkdir_cmd = f"adb shell mkdir -p {device_path}"
            subprocess.run(mkdir_cmd, shell=True, check=True)

            # Push files
            for file_path in files_to_push:
                if Path(file_path).exists():
                    push_cmd = f"adb push {file_path} {device_path}/"
                    result = subprocess.run(push_cmd, shell=True, capture_output=True, text=True)

                    if result.returncode != 0:
                        print(f"❌ Failed to push {file_path}: {result.stderr}")
                        return False
                    else:
                        print(f"   ✅ Pushed {Path(file_path).name}")
                else:
                    print(f"⚠️  File not found: {file_path}")

            # Set executable permissions
            chmod_cmd = f"adb shell chmod +x {device_path}/executorch_runner_integrated"
            subprocess.run(chmod_cmd, shell=True, check=True)

            print("✅ Push complete")
            return True

        except Exception as e:
            print(f"❌ Push error: {e}")
            return False

    def _setup_device_environment(self) -> bool:
        """Setup device environment for testing"""
        print("🔧 Setting up device environment...")

        try:
            # Verify Vulkan support
            vulkan_cmd = "adb shell ls /vendor/lib64/libvulkan.so"
            vulkan_result = subprocess.run(vulkan_cmd, shell=True, capture_output=True, text=True)

            if vulkan_result.returncode != 0:
                print("⚠️  Vulkan support not verified on device")

            # Check ION memory
            ion_cmd = "adb shell ls /dev/ion"
            ion_result = subprocess.run(ion_cmd, shell=True, capture_output=True, text=True)

            if ion_result.returncode == 0:
                print("✅ ION memory device available")
            else:
                print("⚠️  ION memory device not found")

            # Test basic runner execution
            test_cmd = "adb shell /data/local/tmp/neural_interposer_phase3/executorch_runner_integrated --help"
            test_result = subprocess.run(test_cmd, shell=True, capture_output=True, text=True, timeout=10)

            if test_result.returncode == 0:
                print("✅ Runner executable verified")
                return True
            else:
                print(f"❌ Runner test failed: {test_result.stderr}")
                return False

        except Exception as e:
            print(f"❌ Environment setup error: {e}")
            return False

    def run_latency_tests(self) -> Dict[str, Any]:
        """Run end-to-end latency tests (<300ms target)"""
        print("⏱️  Running latency verification tests...")

        latency_results = {
            'sequences_tested': [],
            'average_latencies': [],
            'min_latencies': [],
            'max_latencies': [],
            'target_met': False,
            'falsification_findings': []
        }

        config = self.test_configs['latency_test']

        for seq_len in config['sequences']:
            print(f"   Testing sequence length: {seq_len}")

            latencies = []

            for run in range(config['runs_per_config']):
                latency = self._run_single_latency_test(seq_len)
                if latency > 0:
                    latencies.append(latency)
                    print(".1f")
                else:
                    print(f"   Run {run + 1}: Failed")

            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                min_latency = min(latencies)
                max_latency = max(latencies)

                latency_results['sequences_tested'].append(seq_len)
                latency_results['average_latencies'].append(avg_latency)
                latency_results['min_latencies'].append(min_latency)
                latency_results['max_latencies'].append(max_latency)

                print(".1f")
                # Check against target (scale target for sequence length)
                scaled_target = config['target_latency_ms'] * (seq_len / 256)  # Normalize to 256
                if avg_latency > scaled_target:
                    latency_results['falsification_findings'].append({
                        'sequence': seq_len,
                        'average_latency': avg_latency,
                        'target': scaled_target,
                        'excess_ms': avg_latency - scaled_target,
                        'finding': 'Latency exceeds target'
                    })

        # Overall assessment
        if latency_results['average_latencies']:
            overall_avg = sum(latency_results['average_latencies']) / len(latency_results['average_latencies'])
            latency_results['target_met'] = overall_avg < config['target_latency_ms']

            print(".1f")
        return latency_results

    def _run_single_latency_test(self, seq_len: int) -> float:
        """Run single latency test and return execution time"""
        try:
            # Generate test input
            test_input = f"--input_tokens={','.join([str(i % 32000) for i in range(seq_len)])}"

            # Run inference
            cmd = f"adb shell /data/local/tmp/neural_interposer_phase3/executorch_runner_integrated {test_input} --benchmark"
            start_time = time.time()

            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)

            end_time = time.time()
            execution_time = (end_time - start_time) * 1000  # Convert to ms

            if result.returncode == 0:
                # Parse output for actual inference time
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if 'inference_time_ms' in line:
                        try:
                            return float(line.split(':')[1].strip())
                        except:
                            pass

                # Fallback: use total execution time
                return execution_time
            else:
                print(f"   Test failed: {result.stderr}")
                return -1

        except Exception as e:
            print(f"   Test error: {e}")
            return -1

    def run_memory_tests(self) -> Dict[str, Any]:
        """Run memory usage tests (<280MB target)"""
        print("💾 Running memory verification tests...")

        memory_results = {
            'peak_memory_mb': 0,
            'average_memory_mb': 0,
            'memory_pressure_events': 0,
            'target_met': False,
            'falsification_findings': []
        }

        config = self.test_configs['memory_test']

        try:
            # Run memory monitoring test
            mem_cmd = "adb shell /data/local/tmp/neural_interposer_phase3/executorch_runner_integrated --memory_monitor"
            result = subprocess.run(mem_cmd, shell=True, capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                # Parse memory output
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if 'peak_memory_mb' in line:
                        try:
                            peak_mb = float(line.split(':')[1].strip().replace('MB', ''))
                            memory_results['peak_memory_mb'] = peak_mb
                        except:
                            pass
                    elif 'average_memory_mb' in line:
                        try:
                            avg_mb = float(line.split(':')[1].strip().replace('MB', ''))
                            memory_results['average_memory_mb'] = avg_mb
                        except:
                            pass

            # Memory pressure test
            if config['memory_pressure_test']:
                pressure_results = self._run_memory_pressure_test()
                memory_results.update(pressure_results)

            # Check against target
            memory_results['target_met'] = memory_results['peak_memory_mb'] < config['target_memory_mb']

            print(".1f"            print(".1f"
            if not memory_results['target_met']:
                excess = memory_results['peak_memory_mb'] - config['target_memory_mb']
                memory_results['falsification_findings'].append({
                    'peak_memory': memory_results['peak_memory_mb'],
                    'target': config['target_memory_mb'],
                    'excess_mb': excess,
                    'finding': 'Memory usage exceeds target'
                })

        except Exception as e:
            print(f"❌ Memory test error: {e}")

        return memory_results

    def _run_memory_pressure_test(self) -> Dict[str, Any]:
        """Test memory pressure handling"""
        pressure_results = {'memory_pressure_events': 0}

        try:
            # Run test with memory pressure
            pressure_cmd = "adb shell /data/local/tmp/neural_interposer_phase3/executorch_runner_integrated --memory_pressure_test"
            result = subprocess.run(pressure_cmd, shell=True, capture_output=True, text=True, timeout=120)

            if result.returncode == 0:
                # Parse pressure test results
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if 'pressure_events' in line:
                        try:
                            pressure_results['memory_pressure_events'] = int(line.split(':')[1].strip())
                        except:
                            pass

        except Exception as e:
            print(f"Memory pressure test error: {e}")

        return pressure_results

    def run_accuracy_tests(self) -> Dict[str, Any]:
        """Run accuracy validation tests (>99% target)"""
        print("🎯 Running accuracy verification tests...")

        accuracy_results = {
            'cosine_similarity': 0.0,
            'perplexity_ratio': 0.0,
            'accuracy_degradation_percent': 0.0,
            'target_met': False,
            'falsification_findings': []
        }

        config = self.test_configs['accuracy_test']

        try:
            # Run accuracy comparison test
            accuracy_cmd = "adb shell /data/local/tmp/neural_interposer_phase3/executorch_runner_integrated --accuracy_test"
            result = subprocess.run(accuracy_cmd, shell=True, capture_output=True, text=True, timeout=120)

            if result.returncode == 0:
                # Parse accuracy results
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if 'cosine_similarity' in line:
                        try:
                            accuracy_results['cosine_similarity'] = float(line.split(':')[1].strip())
                        except:
                            pass
                    elif 'perplexity_ratio' in line:
                        try:
                            accuracy_results['perplexity_ratio'] = float(line.split(':')[1].strip())
                        except:
                            pass

                # Calculate degradation
                accuracy_results['accuracy_degradation_percent'] = (1 - accuracy_results['cosine_similarity']) * 100
                accuracy_results['target_met'] = accuracy_results['cosine_similarity'] > config['accuracy_threshold']

                print(".4f")
                print(".2f")
                print(f"   Target met: {'✅' if accuracy_results['target_met'] else '❌'}")

                if not accuracy_results['target_met']:
                    degradation = accuracy_results['accuracy_degradation_percent']
                    target_degradation = (1 - config['accuracy_threshold']) * 100
                    excess_degradation = degradation - target_degradation

                    accuracy_results['falsification_findings'].append({
                        'cosine_similarity': accuracy_results['cosine_similarity'],
                        'target': config['accuracy_threshold'],
                        'degradation_percent': degradation,
                        'excess_degradation': excess_degradation,
                        'finding': 'Accuracy below target threshold'
                    })

        except Exception as e:
            print(f"❌ Accuracy test error: {e}")

        return accuracy_results

    def run_hardware_tests(self) -> Dict[str, Any]:
        """Run hardware utilization tests"""
        print("🔧 Running hardware utilization tests...")

        hardware_results = {
            'gpu_utilization_percent': 0,
            'temperature_c': 0,
            'power_draw_mw': 0,
            'ion_efficiency_percent': 0,
            'targets_met': False,
            'falsification_findings': []
        }

        config = self.test_configs['hardware_test']

        try:
            # Run hardware monitoring test
            hw_cmd = "adb shell /data/local/tmp/neural_interposer_phase3/executorch_runner_integrated --hardware_monitor"
            result = subprocess.run(hw_cmd, shell=True, capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                # Parse hardware results
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if 'gpu_utilization' in line:
                        try:
                            hardware_results['gpu_utilization_percent'] = float(line.split(':')[1].strip().replace('%', ''))
                        except:
                            pass
                    elif 'temperature' in line:
                        try:
                            hardware_results['temperature_c'] = float(line.split(':')[1].strip().replace('°C', ''))
                        except:
                            pass
                    elif 'power_draw' in line:
                        try:
                            hardware_results['power_draw_mw'] = float(line.split(':')[1].strip().replace('mW', ''))
                        except:
                            pass
                    elif 'ion_efficiency' in line:
                        try:
                            hardware_results['ion_efficiency_percent'] = float(line.split(':')[1].strip().replace('%', ''))
                        except:
                            pass

            # Check targets
            gpu_ok = hardware_results['gpu_utilization_percent'] > config['gpu_utilization_target']
            temp_ok = hardware_results['temperature_c'] < config['thermal_limit_c']
            power_ok = hardware_results['power_draw_mw'] < config['power_budget_mw']

            hardware_results['targets_met'] = gpu_ok and temp_ok and power_ok

            print(".1f")
            print(".1f")
            print(".0f")
            print(f"   All targets met: {'✅' if hardware_results['targets_met'] else '❌'}")

            # Record falsification findings
            if not gpu_ok:
                hardware_results['falsification_findings'].append({
                    'metric': 'gpu_utilization',
                    'actual': hardware_results['gpu_utilization_percent'],
                    'target': config['gpu_utilization_target'],
                    'finding': 'GPU utilization below target'
                })

            if not temp_ok:
                hardware_results['falsification_findings'].append({
                    'metric': 'temperature',
                    'actual': hardware_results['temperature_c'],
                    'target': config['thermal_limit_c'],
                    'finding': 'Temperature exceeds limit'
                })

            if not power_ok:
                hardware_results['falsification_findings'].append({
                    'metric': 'power_draw',
                    'actual': hardware_results['power_draw_mw'],
                    'target': config['power_budget_mw'],
                    'finding': 'Power draw exceeds budget'
                })

        except Exception as e:
            print(f"❌ Hardware test error: {e}")

        return hardware_results

    def run_complete_verification(self) -> Dict[str, Any]:
        """
        Run complete Phase 3 verification suite

        Returns:
            Comprehensive verification results
        """
        print("🔬 PHASE 3 DEVICE VERIFICATION: Falsification Testing")
        print("=" * 60)

        verification_results = {
            'timestamp': time.time(),
            'phase': 'Phase 3 Week 7',
            'description': 'Integrated LFM Pipeline Device Verification',
            'device_connection': False,
            'deployment': False,
            'latency_tests': {},
            'memory_tests': {},
            'accuracy_tests': {},
            'hardware_tests': {},
            'overall_success': False,
            'falsification_report': [],
            'recommendations': []
        }

        # 1. Connect to device
        verification_results['device_connection'] = self.connect_to_device()
        if not verification_results['device_connection']:
            verification_results['falsification_report'].append({
                'severity': 'CRITICAL',
                'finding': 'Cannot connect to MediaTek device',
                'impact': 'Unable to verify Phase 3 claims on target hardware'
            })
            return verification_results

        # 2. Deploy pipeline
        verification_results['deployment'] = self.deploy_pipeline()
        if not verification_results['deployment']:
            verification_results['falsification_report'].append({
                'severity': 'CRITICAL',
                'finding': 'Pipeline deployment failed',
                'impact': 'Cannot test integrated LFM pipeline on device'
            })
            return verification_results

        # 3. Run all tests
        print("\n🧪 Running verification test suite...")
        verification_results['latency_tests'] = self.run_latency_tests()
        verification_results['memory_tests'] = self.run_memory_tests()
        verification_results['accuracy_tests'] = self.run_accuracy_tests()
        verification_results['hardware_tests'] = self.run_hardware_tests()

        # 4. Analyze results and generate falsification report
        self._analyze_results(verification_results)

        # 5. Generate recommendations
        self._generate_recommendations(verification_results)

        # 6. Overall assessment
        verification_results['overall_success'] = self._assess_overall_success(verification_results)

        # Save results
        self._save_verification_results(verification_results)

        print("\n🎯 VERIFICATION COMPLETE")
        print(f"Overall Success: {'✅' if verification_results['overall_success'] else '❌'}")
        print(f"Falsification Findings: {len(verification_results['falsification_report'])}")
        print(f"Recommendations: {len(verification_results['recommendations'])}")

        return verification_results

    def _analyze_results(self, results: Dict[str, Any]):
        """Analyze test results and generate falsification findings"""
        print("\n📊 Analyzing results...")
        falsification_report = []

        # Latency analysis
        latency = results['latency_tests']
        if latency.get('falsification_findings'):
            for finding in latency['falsification_findings']:
                falsification_report.append({
                    'severity': 'HIGH' if finding['excess_ms'] > 100 else 'MEDIUM',
                    'category': 'Performance',
                    'finding': f"Latency {finding['excess_ms']:.1f}ms over target for seq_len {finding['sequence']}",
                    'impact': 'May not meet <300ms end-to-end requirement',
                    'recommendations': ['Optimize TriX kernel execution', 'Improve prefetching accuracy', 'Reduce ION memory latency']
                })

        # Memory analysis
        memory = results['memory_tests']
        if memory.get('falsification_findings'):
            for finding in memory['falsification_findings']:
                falsification_report.append({
                    'severity': 'HIGH',
                    'category': 'Memory',
                    'finding': f"Memory usage {finding['excess_mb']:.1f}MB over 280MB target",
                    'impact': 'May cause OOM on memory-constrained devices',
                    'recommendations': ['Optimize ION allocation strategy', 'Improve layer eviction policy', 'Reduce prefetch buffer size']
                })

        # Accuracy analysis
        accuracy = results['accuracy_tests']
        if accuracy.get('falsification_findings'):
            for finding in accuracy['falsification_findings']:
                falsification_report.append({
                    'severity': 'CRITICAL',
                    'category': 'Accuracy',
                    'finding': f"Accuracy {finding['degradation_percent']:.2f}% below 99% threshold",
                    'impact': 'Unacceptable accuracy degradation for LFM inference',
                    'recommendations': ['Improve quantization calibration', 'Fix numerical precision issues', 'Validate TriX computation accuracy']
                })

        # Hardware analysis
        hardware = results['hardware_tests']
        if hardware.get('falsification_findings'):
            for finding in hardware['falsification_findings']:
                severity = 'HIGH' if finding['metric'] == 'temperature' else 'MEDIUM'
                falsification_report.append({
                    'severity': severity,
                    'category': 'Hardware',
                    'finding': f"{finding['metric']} constraint violated",
                    'impact': 'May cause thermal throttling or battery drain',
                    'recommendations': ['Optimize Mali GPU kernels', 'Implement thermal-aware scheduling', 'Reduce power consumption']
                })

        results['falsification_report'] = falsification_report

        print(f"   Falsification findings: {len(falsification_report)}")

        for finding in falsification_report[:3]:  # Show first 3
            print(f"   {finding['severity']}: {finding['finding']}")

    def _generate_recommendations(self, results: Dict[str, Any]):
        """Generate actionable recommendations based on findings"""
        recommendations = []

        # Performance recommendations
        if not results['latency_tests'].get('target_met', False):
            recommendations.extend([
                "Phase 3 Week 8: Optimize TriX kernel execution for Mali-G52",
                "Implement adaptive prefetching based on I/O patterns",
                "Profile and optimize ION memory access latency"
            ])

        # Memory recommendations
        if not results['memory_tests'].get('target_met', False):
            recommendations.extend([
                "Implement ION memory pool optimization",
                "Improve layer sharding eviction policy",
                "Reduce prefetching memory overhead"
            ])

        # Accuracy recommendations
        if not results['accuracy_tests'].get('target_met', False):
            recommendations.extend([
                "Fix numerical precision issues in quantized operations",
                "Improve quantization calibration for LFM2-350M",
                "Validate TriX computation against CPU reference"
            ])

        # Hardware recommendations
        if not results['hardware_tests'].get('targets_met', False):
            recommendations.extend([
                "Optimize Vulkan compute shaders for Mali-G52",
                "Implement thermal-aware execution scheduling",
                "Profile and reduce power consumption"
            ])

        results['recommendations'] = recommendations

    def _assess_overall_success(self, results: Dict[str, Any]) -> bool:
        """Assess overall verification success"""
        latency_ok = results['latency_tests'].get('target_met', False)
        memory_ok = results['memory_tests'].get('target_met', False)
        accuracy_ok = results['accuracy_tests'].get('target_met', False)
        hardware_ok = results['hardware_tests'].get('targets_met', False)

        # All critical targets must be met
        return latency_ok and memory_ok and accuracy_ok and hardware_ok

    def _save_verification_results(self, results: Dict[str, Any]):
        """Save verification results to file"""
        output_file = Path("research/brack/phase3_device_verification_results.json")

        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            print(f"💾 Results saved to {output_file}")

        except Exception as e:
            print(f"❌ Failed to save results: {e}")

def main():
    """Main verification function"""
    print("🔬 Phase 3 Device Verification: Falsification Testing")
    print("=" * 60)

    # Initialize verifier
    verifier = Phase3DeviceVerifier()

    # Run complete verification
    results = verifier.run_complete_verification()

    # Print summary
    print("\n🎯 VERIFICATION SUMMARY")
    print(f"Device Connection: {'✅' if results['device_connection'] else '❌'}")
    print(f"Pipeline Deployment: {'✅' if results['deployment'] else '❌'}")

    if results['device_connection'] and results['deployment']:
        print(f"Latency Target (<300ms): {'✅' if results['latency_tests'].get('target_met') else '❌'}")
        print(f"Memory Target (<280MB): {'✅' if results['memory_tests'].get('target_met') else '❌'}")
        print(f"Accuracy Target (>99%): {'✅' if results['accuracy_tests'].get('target_met') else '❌'}")
        print(f"Hardware Targets: {'✅' if results['hardware_tests'].get('targets_met') else '❌'}")

        print("\n📋 FALSIFICATION FINDINGS:")
        for i, finding in enumerate(results.get('falsification_report', [])[:5], 1):
            print(f"   {i}. [{finding['severity']}] {finding['finding']}")

        print("\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(results.get('recommendations', [])[:5], 1):
            print(f"   {i}. {rec}")

    print("\n🏁 VERIFICATION COMPLETE")
    print(f"Overall Result: {'✅ SUCCESS' if results['overall_success'] else '❌ NEEDS OPTIMIZATION'}")

    if not results['overall_success']:
        print("\n🔧 Proceed to Phase 3 Week 8: Performance Optimization")
        print("   - Address falsification findings")
        print("   - Implement recommendations")
        print("   - Optimize for MediaTek + Mali hardware")

if __name__ == "__main__":
    main()