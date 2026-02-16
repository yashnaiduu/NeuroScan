#!/usr/bin/env python3
"""Download and setup Kaggle brain tumor MRI dataset."""

import kagglehub
import os
import shutil

# Download latest version
print("Downloading brain tumor MRI dataset from Kaggle...")
path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")

print(f"Dataset downloaded to: {path}")

# The dataset should have Training and Testing folders
# Let's create symlink to it
dataset_link = "./Dataset"

if os.path.exists(dataset_link):
    if os.path.islink(dataset_link):
        os.unlink(dataset_link)
    else:
        shutil.rmtree(dataset_link)

os.symlink(path, dataset_link)
print(f"Created symlink: {dataset_link} -> {path}")

# Verify structure
print("\nDataset structure:")
for root, dirs, files in os.walk(dataset_link):
    level = root.replace(dataset_link, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    if level < 2:  # Only show first 2 levels
        subindent = ' ' * 2 * (level + 1)
        for file in files[:3]:  # Show first 3 files
            print(f'{subindent}{file}')
        if len(files) > 3:
            print(f'{subindent}... and {len(files) - 3} more files')

print("\n✅ Dataset ready for use!")
