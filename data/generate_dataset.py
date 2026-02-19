"""
Generate synthetic spectral dataset CSV file
Creates a comprehensive dataset for training machine learning models
"""

import csv
import random
import os

def gauss_random(mean=0, std=1):
    """Generate a random number from normal distribution using Box-Muller transform"""
    u1 = random.random()
    u2 = random.random()
    z0 = (-2 * (u1 ** 0.0001 if u1 == 0 else u1) ** 0.5) * (2 * 3.14159265359 * u2) ** 0.5
    import math
    return mean + std * math.sqrt(-2 * math.log(u1 if u1 > 0 else 0.0001)) * math.cos(2 * math.pi * u2)

def clip(value, min_val=0, max_val=1):
    """Clip value to range [min_val, max_val]"""
    return max(min_val, min(max_val, value))

def generate_synthetic_dataset(samples_per_category=200, output_file='synthetic_data.csv'):
    """
    Generate synthetic spectral dataset for three fruits with organic/non-organic samples.
    
    Args:
        samples_per_category: Number of samples to generate per fruit per category
        output_file: Output CSV filename
    """
    random.seed(42)
    
    # Base spectral signatures for each fruit (8 channels: F1-F8)
    # These represent normalized reflectance values from optical spectroscopy
    fruit_signatures = {
        'Apple': [0.45, 0.52, 0.58, 0.62, 0.55, 0.48, 0.42, 0.38],
        'Banana': [0.72, 0.78, 0.82, 0.85, 0.80, 0.75, 0.68, 0.62],
        'Tomato': [0.68, 0.42, 0.35, 0.38, 0.45, 0.52, 0.48, 0.44]
    }
    
    # Organic spectral shift patterns (slight variations in organic samples)
    # Organic produce often shows different reflectance due to soil nutrients
    organic_shifts = {
        'Apple': [0.02, 0.03, 0.025, 0.02, 0.015, 0.02, 0.025, 0.03],
        'Banana': [0.015, 0.02, 0.025, 0.03, 0.025, 0.02, 0.015, 0.01],
        'Tomato': [0.025, 0.02, 0.015, 0.02, 0.025, 0.03, 0.028, 0.022]
    }
    
    data = []
    
    for fruit_name, base_signature in fruit_signatures.items():
        # Generate Non-Organic samples
        for sample_id in range(samples_per_category):
            # Add random noise to base signature
            noise = [gauss_random(0, 0.025) for _ in range(8)]
            spectral_data = [clip(base_signature[i] + noise[i]) for i in range(8)]
            
            sample = {
                'Sample_ID': f'{fruit_name}_NonOrg_{sample_id:03d}',
                'F1': round(spectral_data[0], 6),
                'F2': round(spectral_data[1], 6),
                'F3': round(spectral_data[2], 6),
                'F4': round(spectral_data[3], 6),
                'F5': round(spectral_data[4], 6),
                'F6': round(spectral_data[5], 6),
                'F7': round(spectral_data[6], 6),
                'F8': round(spectral_data[7], 6),
                'Fruit': fruit_name,
                'Organic': 'Non-Organic'
            }
            data.append(sample)
        
        # Generate Organic samples with spectral shift
        for sample_id in range(samples_per_category):
            # Add organic shift and random noise
            noise = [gauss_random(0, 0.025) for _ in range(8)]
            spectral_data = [clip(base_signature[i] + organic_shifts[fruit_name][i] + noise[i]) for i in range(8)]
            
            sample = {
                'Sample_ID': f'{fruit_name}_Org_{sample_id:03d}',
                'F1': round(spectral_data[0], 6),
                'F2': round(spectral_data[1], 6),
                'F3': round(spectral_data[2], 6),
                'F4': round(spectral_data[3], 6),
                'F5': round(spectral_data[4], 6),
                'F6': round(spectral_data[5], 6),
                'F7': round(spectral_data[6], 6),
                'F8': round(spectral_data[7], 6),
                'Fruit': fruit_name,
                'Organic': 'Organic'
            }
            data.append(sample)
    
    # Shuffle the dataset
    random.shuffle(data)
    
    # Save to CSV
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, output_file)
    
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['Sample_ID', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'Fruit', 'Organic']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"✓ Dataset generated successfully!")
    print(f"✓ Saved to: {output_path}")
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(data)}")
    print(f"  Fruit distribution:")
    fruit_counts = {}
    organic_counts = {}
    for row in data:
        fruit_counts[row['Fruit']] = fruit_counts.get(row['Fruit'], 0) + 1
        organic_counts[row['Organic']] = organic_counts.get(row['Organic'], 0) + 1
    
    for fruit, count in sorted(fruit_counts.items()):
        print(f"    - {fruit}: {count} samples")
    print(f"\n  Organic distribution:")
    for status, count in sorted(organic_counts.items()):
        print(f"    - {status}: {count} samples")
    print(f"\n  Spectral channels: F1, F2, F3, F4, F5, F6, F7, F8")
    print(f"  Sample ID format: [Fruit]_[Org/NonOrg]_[Number]")
    
    return data

if __name__ == '__main__':
    print("="*60)
    print("Synthetic Spectral Dataset Generator")
    print("="*60)
    print()
    generate_synthetic_dataset(samples_per_category=200)
    print("\n" + "="*60)
