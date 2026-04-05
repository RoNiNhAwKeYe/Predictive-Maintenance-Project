"""
Audio Sensor Simulator
Simulates acoustic bearing sensor data for testing
Can be used to generate training data or test the system
"""

import numpy as np
from scipy import signal
from scipy.io import wavfile
import os

class AcousticSensorSimulator:
    """Simulate bearing fault acoustic signatures"""
    
    def __init__(self, sr=20000):
        """
        Initialize simulator
        Args:
            sr: Sampling rate in Hz
        """
        self.sr = sr
    
    def generate_healthy_bearing(self, duration=1.0):
        """
        Generate acoustic signature of healthy bearing
        Clean signal with low amplitude and minimal harmonics
        
        Args:
            duration: Duration in seconds
            
        Returns:
            numpy array of audio samples
        """
        samples = int(duration * self.sr)
        t = np.linspace(0, duration, samples, False)
        
        # Healthy bearing: dominant frequency around 2 kHz
        dominant_freq = 2000
        signal_clean = 0.1 * np.sin(2 * np.pi * dominant_freq * t)
        
        # Add minimal harmonic
        harmonic = 0.02 * np.sin(2 * np.pi * dominant_freq * 2 * t)
        
        # Add light noise
        noise = 0.02 * np.random.randn(samples)
        
        audio = signal_clean + harmonic + noise
        audio = np.clip(audio, -1.0, 1.0)
        
        return audio.astype(np.float32)
    
    def generate_faulty_bearing(self, duration=1.0, fault_severity=0.7):
        """
        Generate acoustic signature of faulty bearing
        Multiple frequencies, impulses, and increased noise
        
        Args:
            duration: Duration in seconds
            fault_severity: Fault severity (0.0 to 1.0)
                - 0.0: Barely faulty
                - 0.5: Moderate fault
                - 1.0: Severe fault
                
        Returns:
            numpy array of audio samples
        """
        samples = int(duration * self.sr)
        t = np.linspace(0, duration, samples, False)
        
        # Base frequencies
        base_freq = 1500
        harmonic_freq = 3000
        
        # Signal components increase with severity
        base_amplitude = 0.15 + 0.2 * fault_severity
        harmonic_amplitude = 0.1 + 0.15 * fault_severity
        
        base = base_amplitude * np.sin(2 * np.pi * base_freq * t)
        harmonic = harmonic_amplitude * np.sin(2 * np.pi * harmonic_freq * t)
        
        # Characteristic bearing impulses (spalling)
        impulse_freq = 100 + int(100 * fault_severity)  # Impulses per second
        impulses = np.zeros(samples)
        
        for impulse_time in np.arange(0, duration, 1/impulse_freq):
            impulse_idx = int(impulse_time * self.sr)
            if impulse_idx < samples:
                impulse_width = int((0.001 + 0.002 * fault_severity) * self.sr)
                impulses[impulse_idx:min(impulse_idx + impulse_width, samples)] += (
                    0.4 * fault_severity
                )
        
        # Increased noise with fault
        noise_level = 0.03 + 0.05 * fault_severity
        noise = noise_level * np.random.randn(samples)
        
        # Combine all components
        audio = base + harmonic + impulses + noise
        audio = np.clip(audio, -1.0, 1.0)
        
        return audio.astype(np.float32)
    
    def save_wav(self, audio, filename, sr=None):
        """
        Save audio array to WAV file
        
        Args:
            audio: Audio samples (float32, [-1, 1] range)
            filename: Output filename
            sr: Sampling rate (uses self.sr if None)
        """
        if sr is None:
            sr = self.sr
        
        # Convert float32 [-1, 1] to int16
        audio_int16 = np.int16(audio * 32767)
        
        wavfile.write(filename, sr, audio_int16)
        print(f"✓ Saved: {filename}")
    
    def generate_dataset(self, output_dir='test_audio', num_healthy=10, num_faulty=10):
        """
        Generate dataset of test audio files
        
        Args:
            output_dir: Output directory for audio files
            num_healthy: Number of healthy bearing samples
            num_faulty: Number of faulty bearing samples
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nGenerating test dataset in '{output_dir}'...")
        print(f"  Healthy samples: {num_healthy}")
        print(f"  Faulty samples: {num_faulty}")
        print()
        
        # Generate healthy samples
        for i in range(num_healthy):
            audio = self.generate_healthy_bearing(duration=0.5)
            filename = os.path.join(output_dir, f'healthy_{i:03d}.wav')
            self.save_wav(audio, filename)
        
        # Generate faulty samples (varying severity)
        for i in range(num_faulty):
            severity = (i + 1) / num_faulty  # Gradually increase severity
            audio = self.generate_faulty_bearing(duration=0.5, fault_severity=severity)
            filename = os.path.join(output_dir, f'faulty_{i:03d}.wav')
            self.save_wav(audio, filename)
        
        print(f"\n✓ Dataset generated successfully!")
        print(f"  Total files: {num_healthy + num_faulty}")
        print(f"  Location: {os.path.abspath(output_dir)}")

def main():
    """Generate test dataset"""
    simulator = AcousticSensorSimulator(sr=20000)
    
    # Generate a small dataset
    simulator.generate_dataset(
        output_dir='test_audio',
        num_healthy=5,
        num_faulty=5
    )
    
    # Generate single example files
    print("\nGenerating example files...")
    healthy = simulator.generate_healthy_bearing(duration=0.5)
    simulator.save_wav(healthy, 'example_healthy.wav')
    
    faulty = simulator.generate_faulty_bearing(duration=0.5, fault_severity=1.0)
    simulator.save_wav(faulty, 'example_faulty.wav')

if __name__ == '__main__':
    main()
