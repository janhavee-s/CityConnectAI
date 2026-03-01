import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.mineral_data import get_combined_data

df = get_combined_data(state_level=True)
df.to_csv("state_level_mineral_dataset_2014_2025.csv", index=False)

print("State-level dataset exported.")