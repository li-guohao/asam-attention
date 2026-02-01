"""
Robustness Testing for ASAM
============================

Tests algorithm robustness against:
1. Adversarial inputs
2. Gradient stability
3. Numerical precision
4. Variable sequence lengths
5. Edge cases
6. Noise resilience
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from asam import ASAMLayer, ASAMConfig


class RobustnessTester:
    """Comprehensive robustness testing suite."""
    
    def __init__(self, dim: int = 256, num_heads: int = 4, seq_len: int = 512):
        self.dim = dim
        self.num_heads = num_heads
        self.seq_len = seq_len
        
        config = ASAMConfig(
            dim=dim,
            num_heads=num_heads,
            pattern_type="hierarchical",
            use_adaptive_gate=True,
        )
        self.model = ASAMLayer(config)
        self.model.eval()
    
    def test_gradient_stability(self, num_trials: int = 100) -> Dict:
        """
        Test gradient stability with different random inputs.
        Checks for vanishing/exploding gradients.
        """
        print("\n" + "="*60)
        print("Testing Gradient Stability...")
        print("="*60)
        
        grad_norms = []
        has_nan = []
        has_inf = []
        
        for i in range(num_trials):
            # Random input with different scales
            scale = 10 ** np.random.uniform(-3, 3)
            x = torch.randn(2, self.seq_len, self.dim) * scale
            x.requires_grad = True
            
            self.model.train()
            output, _ = self.model(x)
            loss = output.sum()
            loss.backward()
            
            grad_norm = x.grad.norm().item()
            grad_norms.append(grad_norm)
            has_nan.append(torch.isnan(x.grad).any().item())
            has_inf.append(torch.isinf(x.grad).any().item())
            
            self.model.zero_grad()
        
        results = {
            'mean_grad_norm': np.mean(grad_norms),
            'std_grad_norm': np.std(grad_norms),
            'min_grad_norm': np.min(grad_norms),
            'max_grad_norm': np.max(grad_norms),
            'nan_rate': np.mean(has_nan),
            'inf_rate': np.mean(has_inf),
        }
        
        print(f"Gradient norm: {results['mean_grad_norm']:.4f} ± {results['std_grad_norm']:.4f}")
        print(f"Range: [{results['min_grad_norm']:.4e}, {results['max_grad_norm']:.4e}]")
        print(f"NaN rate: {results['nan_rate']:.2%}")
        print(f"Inf rate: {results['inf_rate']:.2%}")
        
        # Check for vanishing/exploding
        if results['mean_grad_norm'] < 1e-6:
            print("⚠️ WARNING: Potential vanishing gradients!")
        if results['mean_grad_norm'] > 1e3:
            print("⚠️ WARNING: Potential exploding gradients!")
        if results['nan_rate'] > 0:
            print("❌ FAILED: NaN gradients detected!")
        
        return results
    
    def test_numerical_precision(self) -> Dict:
        """
        Test numerical precision with different dtypes.
        """
        print("\n" + "="*60)
        print("Testing Numerical Precision...")
        print("="*60)
        
        dtypes = [torch.float32, torch.float64]
        if torch.cuda.is_available():
            dtypes.append(torch.float16)
        
        x = torch.randn(2, self.seq_len, self.dim)
        
        results = {}
        outputs = {}
        
        for dtype in dtypes:
            try:
                model = self.model.to(dtype)
                x_typed = x.to(dtype)
                
                with torch.no_grad():
                    output, _ = model(x_typed)
                
                outputs[str(dtype)] = output.float()
                results[str(dtype)] = {
                    'mean': output.mean().item(),
                    'std': output.std().item(),
                    'has_nan': torch.isnan(output).any().item(),
                    'has_inf': torch.isinf(output).any().item(),
                }
                
                status = "✓" if not results[str(dtype)]['has_nan'] else "❌"
                print(f"{status} {dtype}: mean={results[str(dtype)]['mean']:.4f}, "
                      f"std={results[str(dtype)]['std']:.4f}")
            
            except Exception as e:
                print(f"❌ {dtype}: Failed - {e}")
                results[str(dtype)] = {'error': str(e)}
        
        # Check consistency between float32 and float64
        if 'torch.float32' in outputs and 'torch.float64' in outputs:
            diff = (outputs['torch.float32'] - outputs['torch.float64']).abs().mean().item()
            results['f32_f64_diff'] = diff
            print(f"\nFloat32 vs Float64 difference: {diff:.6e}")
            
            if diff > 1e-3:
                print("⚠️ WARNING: Large precision difference!")
        
        return results
    
    def test_variable_sequence_lengths(self, lengths: List[int] = None) -> Dict:
        """
        Test with variable sequence lengths.
        """
        print("\n" + "="*60)
        print("Testing Variable Sequence Lengths...")
        print("="*60)
        
        if lengths is None:
            lengths = [128, 256, 512, 1024, 2048, 4096]
        
        results = {}
        
        for length in lengths:
            try:
                x = torch.randn(2, length, self.dim)
                
                with torch.no_grad():
                    start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                    end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                    
                    if start:
                        start.record()
                    
                    output, info = self.model(x)
                    
                    if end:
                        end.record()
                        torch.cuda.synchronize()
                        time_ms = start.elapsed_time(end)
                    else:
                        time_ms = None
                
                results[length] = {
                    'output_shape': list(output.shape),
                    'has_nan': torch.isnan(output).any().item(),
                    'has_inf': torch.isinf(output).any().item(),
                    'time_ms': time_ms,
                    'sparse_ratio': info.get('sparse_ratio', None) if info else None
                }
                
                status = "✓" if not results[length]['has_nan'] else "❌"
                time_str = f"{time_ms:.2f}ms" if time_ms else "N/A"
                print(f"{status} Length={length:5d}: shape={results[length]['output_shape']}, "
                      f"time={time_str}")
            
            except RuntimeError as e:
                print(f"❌ Length={length}: Failed - {e}")
                results[length] = {'error': str(e)}
        
        return results
    
    def test_noise_resilience(self, noise_levels: List[float] = None) -> Dict:
        """
        Test resilience to input noise.
        """
        print("\n" + "="*60)
        print("Testing Noise Resilience...")
        print("="*60)
        
        if noise_levels is None:
            noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
        
        # Clean input
        x_clean = torch.randn(2, self.seq_len, self.dim)
        
        with torch.no_grad():
            output_clean, _ = self.model(x_clean)
        
        results = {}
        
        for noise_level in noise_levels:
            # Add noise
            noise = torch.randn_like(x_clean) * noise_level
            x_noisy = x_clean + noise
            
            with torch.no_grad():
                output_noisy, _ = self.model(x_noisy)
            
            # Measure change
            relative_change = (output_noisy - output_clean).norm() / output_clean.norm()
            
            results[noise_level] = {
                'relative_change': relative_change.item(),
                'correlation': torch.corrcoef(
                    torch.stack([output_clean.flatten(), output_noisy.flatten()])
                )[0, 1].item()
            }
            
            print(f"Noise={noise_level:.2f}: relative_change={relative_change:.4f}, "
                  f"correlation={results[noise_level]['correlation']:.4f}")
        
        return results
    
    def test_edge_cases(self) -> Dict:
        """
        Test edge cases.
        """
        print("\n" + "="*60)
        print("Testing Edge Cases...")
        print("="*60)
        
        results = {}
        
        test_cases = {
            'all_zeros': torch.zeros(2, self.seq_len, self.dim),
            'all_ones': torch.ones(2, self.seq_len, self.dim),
            'very_large': torch.ones(2, self.seq_len, self.dim) * 1e6,
            'very_small': torch.ones(2, self.seq_len, self.dim) * 1e-6,
            'alternating': torch.tensor([[[1.0, -1.0] * (self.dim // 2)] * self.seq_len] * 2),
            'single_batch': torch.randn(1, self.seq_len, self.dim),
            'minimal_length': torch.randn(2, 2, self.dim),
        }
        
        for name, x in test_cases.items():
            try:
                with torch.no_grad():
                    output, _ = self.model(x)
                
                results[name] = {
                    'success': True,
                    'has_nan': torch.isnan(output).any().item(),
                    'has_inf': torch.isinf(output).any().item(),
                    'mean': output.mean().item(),
                    'std': output.std().item(),
                }
                
                status = "✓" if not results[name]['has_nan'] and not results[name]['has_inf'] else "❌"
                print(f"{status} {name:20s}: mean={results[name]['mean']:12.4e}, "
                      f"std={results[name]['std']:12.4e}")
                
                if results[name]['has_nan']:
                    print(f"   ⚠️ NaN detected!")
                if results[name]['has_inf']:
                    print(f"   ⚠️ Inf detected!")
            
            except Exception as e:
                print(f"❌ {name:20s}: Failed - {e}")
                results[name] = {'success': False, 'error': str(e)}
        
        return results
    
    def test_adversarial_robustness(self, epsilon: float = 0.01, num_steps: int = 10) -> Dict:
        """
        Test robustness to adversarial perturbations (FGSM).
        """
        print("\n" + "="*60)
        print("Testing Adversarial Robustness (FGSM)...")
        print("="*60)
        
        x = torch.randn(2, self.seq_len, self.dim, requires_grad=True)
        
        # Original output
        with torch.no_grad():
            output_orig, _ = self.model(x)
        
        # Generate adversarial example
        self.model.eval()
        x_adv = x.clone().detach().requires_grad_(True)
        
        for step in range(num_steps):
            output_adv, _ = self.model(x_adv)
            
            # Loss: maximize change from original
            loss = -(output_adv - output_orig).norm()
            loss.backward()
            
            # FGSM step
            with torch.no_grad():
                x_adv = x_adv + epsilon * x_adv.grad.sign()
                x_adv = torch.clamp(x_adv, x - epsilon, x + epsilon)
            
            x_adv = x_adv.detach().requires_grad_(True)
            self.model.zero_grad()
        
        # Evaluate
        with torch.no_grad():
            output_final, _ = self.model(x_adv)
        
        relative_change = (output_final - output_orig).norm() / output_orig.norm()
        
        results = {
            'epsilon': epsilon,
            'num_steps': num_steps,
            'relative_change': relative_change.item(),
            'perturbation_norm': (x_adv - x).norm().item(),
        }
        
        print(f"Perturbation norm: {results['perturbation_norm']:.4f}")
        print(f"Output relative change: {results['relative_change']:.4f}")
        
        if results['relative_change'] > 1.0:
            print("⚠️ WARNING: High sensitivity to adversarial perturbations!")
        else:
            print("✓ Good adversarial robustness")
        
        return results
    
    def run_all_tests(self) -> Dict:
        """Run all robustness tests."""
        print("\n" + "="*70)
        print("ASAM ROBUSTNESS TEST SUITE")
        print("="*70)
        
        results = {
            'gradient_stability': self.test_gradient_stability(),
            'numerical_precision': self.test_numerical_precision(),
            'variable_lengths': self.test_variable_sequence_lengths(),
            'noise_resilience': self.test_noise_resilience(),
            'edge_cases': self.test_edge_cases(),
            'adversarial': self.test_adversarial_robustness(),
        }
        
        # Summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        
        all_passed = True
        
        # Check for critical issues
        if results['gradient_stability']['nan_rate'] > 0:
            print("❌ FAILED: NaN gradients detected!")
            all_passed = False
        else:
            print("✓ Gradient stability: PASSED")
        
        if any(r.get('has_nan', False) for r in results['edge_cases'].values() if isinstance(r, dict)):
            print("⚠️ WARNING: NaN in some edge cases")
        else:
            print("✓ Edge cases: PASSED")
        
        print("\nAll tests completed!")
        
        return results


def main():
    """Run robustness tests."""
    tester = RobustnessTester(dim=256, num_heads=4, seq_len=512)
    results = tester.run_all_tests()
    
    # Save results
    import json
    with open('robustness_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\nResults saved to robustness_test_results.json")


if __name__ == "__main__":
    main()
