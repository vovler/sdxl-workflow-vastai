import zipfile_zstd as zipfile

zf = zipfile.ZipFile('/lab/model/unet/unet.zip', 'w', zipfile.ZIP_ZSTANDARD, compresslevel=19)
zf.write('/lab/model/unet/diffusion_pytorch_model.safetensors')