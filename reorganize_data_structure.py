import os
import glob
import shutil
from pathlib import Path
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('avi_reorganization.log')
        ]
    )
    return logging.getLogger(__name__)

def reorganize_avi_files(base_path):
    """
    Reorganizes AVI files from timestamp folders into a single Reorganized_Avis folder.
    
    Args:
        base_path (str): Path to the main directory containing the day folder
    """
    logger = setup_logging()
    base_path = Path(base_path)
    
    # Create Reorganized_Avis folder (removed My_V4_Miniscope subfolder)
    output_dir = base_path / 'Reorganized_Avis'
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")
    
    # Find all AVI files
    pattern = os.path.join(base_path, "**", "My_V4_Miniscope", "*.avi")
    avi_files = glob.glob(pattern, recursive=True)
    
    if not avi_files:
        logger.error(f"No AVI files found in {base_path}")
        return
    
    logger.info(f"Found {len(avi_files)} AVI files")
    
    # Copy and rename files
    for i, avi_path in enumerate(avi_files, 1):
        try:
            # Get timestamp from parent folder name
            timestamp_folder = Path(avi_path).parent.parent.name
            
            # Create new filename with timestamp
            original_filename = Path(avi_path).name
            new_filename = f"{timestamp_folder}_{original_filename}"
            
            # Copy file
            destination = output_dir / new_filename
            shutil.copy2(avi_path, destination)
            
            logger.info(f"[{i}/{len(avi_files)}] Copied: {original_filename} -> {new_filename}")
            
        except Exception as e:
            logger.error(f"Error processing {avi_path}: {str(e)}")

# if __name__ == "__main__":
#     base_path = r"D:\2024.04.25 Ari Punished Fentanyl\Miniscope Data\PunishedFentanyl\3135\T19"
    
#     try:
#         reorganize_avi_files(base_path)
#         print("\nReorganization complete! Check avi_reorganization.log for details.")
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")

def reorganize_specific_animals(root_path):
    """
    Reorganizes AVI files for specific animals and timepoints.
    
    Args:
        root_path (str): Path to the main directory containing animal folders
    """
    logger = setup_logging()
    root_path = Path(root_path)
    
    # Define the animals and their timepoints
    animal_timepoints = {
        '3132': ['T17', 'T21', 'T23', 'T24'],
        '3135': ['T19', 'T21', 'T23', 'T24'],
        '3138': ['T17', 'T21', 'T23'],
        '3139': ['T17', 'T19', 'T21', 'T23', 'T24'],
        '3142': ['T17', 'T19', 'T21', 'T23', 'T24'],
        '3144': ['T17', 'T19', 'T21', 'T23', 'T24'],
        '3146': ['T17', 'T19', 'T21', 'T23', 'T24'],
        '3148': ['T17', 'T19', 'T21', 'T23', 'T24'],
        '3161': ['T17', 'T21', 'T23', 'T24'],
        '3162': ['T19', 'T21', 'T23', 'T24'],
        '3170': ['T19', 'T21', 'T23', 'T24'],
        '3172': ['T19', 'T21', 'T23', 'T24'],
        '3173': ['T19', 'T21', 'T23', 'T24']
    }
    
    # Process each animal and its timepoints
    for animal, timepoints in animal_timepoints.items():
        for timepoint in timepoints:
            try:
                source_path = root_path / animal / timepoint
                
                if not source_path.exists():
                    logger.warning(f"Directory not found: {source_path}")
                    continue
                
                logger.info(f"Processing {animal} - {timepoint}")
                reorganize_avi_files(str(source_path))
                
            except Exception as e:
                logger.error(f"Error processing {animal} {timepoint}: {str(e)}")
    
    logger.info("Completed reorganizing all specified animals and timepoints")

if __name__ == "__main__":
    root_path = r"D:\2024.04.25 Ari Punished Fentanyl\Miniscope Data\PunishedFentanyl"
    
    try:
        reorganize_specific_animals(root_path)
        print("\nReorganization complete! Check avi_reorganization.log for details.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")