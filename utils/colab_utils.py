import shutil
from google.colab import files

def zip_and_download(folder_name, zip_name=None):
    """
    Creates a ZIP archive from a folder and downloads it using Google Colab's file system.

    Parameters:
        folder_name (str): The name of the folder to compress.
        zip_name (str, optional): The name of the resulting ZIP file (without extension). 
                                  If None, it defaults to the folder name.

    Returns:
        None
    """
    
    if zip_name is None:
        zip_name = folder_name

    shutil.make_archive(zip_name, 'zip', folder_name)
    files.download(f"{zip_name}.zip")