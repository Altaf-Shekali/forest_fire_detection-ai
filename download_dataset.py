import kagglehub

path = kagglehub.dataset_download(
    "mohnishsaiprasad/forest-fire-images",
    force_download=True
)

print(path)
