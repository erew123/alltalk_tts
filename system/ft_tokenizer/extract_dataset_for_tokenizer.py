# Simple script to remove LJ Speech formatting for the tokenizer
# Combine metadata_train and metadata_eval.csv into a single file then run
import csv

# Input and output file names
input_file = '/alltalkbeta/metadata_eval.csv'  # combine metadata_train and metadata_eval.csv
output_file = '/alltalkbeta/dataset.txt' # this goes to the tokenizer

# Read the input CSV and write to the output file
with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
     open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    
    # Create CSV reader and writer objects
    reader = csv.reader(infile, delimiter='|')
    writer = csv.writer(outfile, delimiter='|')
    
    # Skip the header
    next(reader, None)
    
    # Process each row
    for row in reader:
        if len(row) >= 2:
            # Write only the second column (index 1) to the output file
            writer.writerow([row[1]])

print(f"Processing complete. Output written to {output_file}")