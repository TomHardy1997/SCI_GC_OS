import openslide
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import argparse
import numpy as np

error_records = []

def get_magnification(image_path):
    try:
        slideimage = openslide.OpenSlide(image_path)
    except:
        print(f'openslide error for {image_path}')
        error_records.append((image_path, 'openslide error'))
        return None

    if 'aperio.AppMag' in slideimage.properties.keys():
        level_0_magnification = int(slideimage.properties['aperio.AppMag'])
    elif 'openslide.mpp-x' in slideimage.properties.keys():
        level_0_magnification = 40 if int(np.floor(float(slideimage.properties['openslide.mpp-x']) * 10)) == 2 else 20
    else:
        print(f'No magnification info for {image_path}, defaulting to 40')
        error_records.append((image_path, 'No magnification info, defaulting to 40'))
        level_0_magnification = 40

    return level_0_magnification

def process_image(row):
    image_path = row.path
    magnification = get_magnification(image_path)
    return magnification

def main(input_csv, output_csv, error_csv, num_threads):
    global error_records
    error_records = []  # Reset error records

    # Read the input CSV file
    df = pd.read_csv(input_csv)
    
    # Use ThreadPoolExecutor for multithreading
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Use tqdm to display a progress bar
        magnifications = list(tqdm(executor.map(process_image, df.itertuples(index=False)), total=len(df)))

    # Add the magnifications to the DataFrame
    df['max_magnification'] = magnifications

    # Save the DataFrame to a new CSV file
    df.to_csv(output_csv, index=False)

    # Save error records to a new CSV file
    if error_records:
        error_df = pd.DataFrame(error_records, columns=['image_path', 'error_message'])
        error_df.to_csv(error_csv, index=False)
    else:
        print('No errors encountered.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract maximum magnification from svs images and save to CSV.")
    parser.add_argument('--input_csv', type=str, required=True, help='Path to the input CSV file.')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to save the output CSV file.')
    parser.add_argument('--error_csv', type=str, required=True, help='Path to save the error CSV file.')
    parser.add_argument('--num_threads', type=int, default=16, help='Number of threads to use for processing.')
    args = parser.parse_args()

    main(args.input_csv, args.output_csv, args.error_csv, args.num_threads)
