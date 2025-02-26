# Name: Muhammad Faheem
# Email: i220485@nu.edu.pk

import os
import re
from tqdm import tqdm



def load_dataset(dataset_dir):
    """
    Loads diary entries from a dataset folder.

    Each file in the dataset_dir is assumed to be a .txt file containing lines of diary entries.
    Each line follows the format: 
      <number>. <diary entry text>
    
    This function extracts the diary entry text by removing the leading numbering and dot.
    
    Args:
        dataset_dir (str): Path to the dataset folder containing .txt files.
    
    Returns:
        list of str: A list containing all diary entries extracted from the files.
    """
    diary_entries = []
    
    # Listing all .txt files in the dataset directory.
    txt_files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.txt')]
    
    for file_path in tqdm(txt_files):
        
        with open(file_path, 'r', encoding='utf-8-sig', errors='replace') as f:
        
            for line in f:
                line = line.strip()
        
                if not line:
                    continue  # Skipping empty lines

                # Using regex to remove the leading number and dot.
                # The pattern matches an optional whitespace, a number, a dot, and the space after it.
                match = re.match(r'^\s*\d+\.\s*(.*)$', line)

                if match:
                    diary_entry = match.group(1)
                else:
                    diary_entry = line  # In case the line doesn't match the expected format.
                
                diary_entries.append(diary_entry)
    
    return diary_entries



# For, testing
if __name__ == '__main__':
    dataset_directory = r"22I-0485_BS-AI-B_NLP-Assignment1\data\dataset"
    
    # Loading the diary entries.
    entries = load_dataset(dataset_directory)
    print("\nTotal diary entries loaded:", len(entries))
    
    # For, testing (as well)
    print("\nSampling the first five elements of the list")
    for i, entry in enumerate(entries[:5]):
        print(f"{i+1}. {entry}")
