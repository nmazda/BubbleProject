import cv2
import os

# Function to resize an image while maintaining aspect ratio
def resize_image(image, target_size):
    height, width = image.shape[:2]
    if height > width:
        scale = target_size / height
    else:
        scale = target_size / width
    resized_image = cv2.resize(image, None, fx=scale, fy=scale)
    return resized_image

# Directory containing original images
input_dir = 'path/to/input' 

# Directory to save cropped squares
output_dir = 'path/to/output' 
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)

        # Manually identified crop boundaries
        left = 774
        upper = 4
        right = 834
        lower = 782
        
        cropped_image = image[upper:lower, left:right]
        # Manually identified crop boundaries
        square_size = 60
        
        height, width = cropped_image.shape[:2]
        num_squares_vertical = height // square_size
        

        cropped_squares = []
        for i in range(num_squares_vertical):
            start_y = i * square_size
            end_y = start_y + square_size
    
            start_x = 0
            while start_x + square_size <= width:
                roi = cropped_image[start_y:end_y, start_x:start_x + square_size]
                resized_square = resize_image(roi, 256)
                cropped_squares.append(resized_square)
                start_x += square_size
        
        base_name = os.path.splitext(filename)[0]
        for idx, square in enumerate(cropped_squares):
            square_name = f'{base_name}_{idx+1}.png'
            save_path = os.path.join(output_dir, square_name)
            cv2.imwrite(save_path, square)
            print(f'Saved: {save_path}')
