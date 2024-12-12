import os
import numpy as np
from PIL import Image

def vertical_merge_image_series(input_directory, output_directory=None):
    """
    Merge all 12 images for each number and plane type.
    
    Parameters:
    - input_directory: Directory containing input images
    - output_directory: Directory to save merged images
    
    Returns:
    - Dictionary of merged image information
    """
    if output_directory is None:
        output_directory = os.path.join(input_directory, 'merged_images')
    os.makedirs(output_directory, exist_ok=True)
    
    merge_results = {}
    
    plane_types = ['xz', 'yz']
    
    numbers = list(range(0, 476, 5))
    
    for num in numbers:
        for plane_type in plane_types:
            matching_images = []
            for suffix in range(12):  # 0 to 11
                filename = f'fl_0_{num:04d}_{suffix}_{plane_type}.png'
                filepath = os.path.join(input_directory, filename)
                
                if os.path.exists(filepath):
                    matching_images.append(filepath)
            
            if len(matching_images) > 0:
                imgs = [Image.open(fp) for fp in matching_images]
                
                assert len(set(img.width for img in imgs)) == 1, f"Width mismatch for {num} {plane_type}"
                
                total_height = sum(img.height for img in imgs)
                
                merged_image = Image.new('RGB', (imgs[0].width, total_height))
                
                current_height = 0
                for img in imgs:
                    merged_image.paste(img, (0, current_height))
                    current_height += img.height
                
                output_filename = f'merged_fl_0_{num:04d}_{plane_type}.png'
                output_filepath = os.path.join(output_directory, output_filename)
                
                merged_image.save(output_filepath)
                
                merge_results[f'{num:04d}_{plane_type}'] = {
                    'output_path': output_filepath,
                    'input_files': [os.path.basename(fp) for fp in matching_images]
                }
                
                print(f"Merged {output_filename}: {len(matching_images)} images")
    
    return merge_results

# Main execution
if __name__ == "__main__":
    # Replace with your actual directory path
    input_directory = "./BW"  # current directory, change as needed
    output_directory = "./mergedBW"
    
    # Merge image series
    results = vertical_merge_image_series(
        input_directory=input_directory,
        output_directory=output_directory
    )
    
    # Print summary
    print("\nMerge Summary:")
    print(f"Total merged images: {len(results)}")
    for key, info in results.items():
        print(f"{key}: {len(info['input_files'])} input files")