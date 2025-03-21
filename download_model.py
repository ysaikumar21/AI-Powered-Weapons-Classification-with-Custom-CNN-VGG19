import gdown

# Google Drive File ID (Replace with your actual ID)
file_id = "1D4fx9bD9t4b-CE2sJDMm0wLLk9g1hyIQ"
output = "vgg19_multilabel_model_sample.h5"  # Save file with this name

# Google Drive Download URL
gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)

print("Download complete!")
