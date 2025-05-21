import os
import numpy as np
import cv2
from src.yolo import predict, get_result, preprocessing, predict_colonies
from src.utils import adjust_mask

def process_images_sequentially(input_dir, output_dir):
    """
    Processes images sequentially by applying the logic from app.py without Streamlit.

    Args:
        input_dir (str): Path to the directory containing input images.
        output_dir (str): Path to the directory where output images will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"processed_{filename}")

            # Read the input image
            image = cv2.imread(input_path)
            if image is None:
                print(f"Failed to read image: {input_path}")
                continue

            # Step 1: Predict wells and masks
            mask, individual_mask, index_mask = predict(image)
            mask_refined = adjust_mask(mask)
            result = get_result(np.float32(mask_refined), image)

            # Step 2: Preprocess the result
            threshold1, threshold2 = 20, 180  # Example thresholds
            processed_result = preprocessing(result, threshold1, threshold2)

            # Step 3: Detect colonies
            colonies, colony_counts, total_colonies, col_img = predict_colonies(image, individual_mask)
            
            '''
            # Save the final result
            cv2.imwrite(output_path, colonies)
            print(f"Processed and saved: {output_path}")

            # Optional: Save individual well images
            for i, well_image in enumerate(col_img):
                well_output_path = os.path.join(output_dir, f"well_{i+1}_{filename}")
                cv2.imwrite(well_output_path, well_image)
                print(f"Saved well {i+1}: {well_output_path}")
            '''

            # Log colony counts
            print(f"Colony counts per well: {colony_counts}")
            print(f"Total colonies: {total_colonies}")

if __name__ == "__main__":
    # Define input and output directories
    input_directory = "/home/carlos/Documents/streamlit_u_net/data"
    output_directory = "/home/carlos/Documents/streamlit_u_net/outputs"

    # Process images sequentially
    process_images_sequentially(input_directory, output_directory)
