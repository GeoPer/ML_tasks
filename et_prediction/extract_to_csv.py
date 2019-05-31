import zipfile

def unzip_file(path_to_zip, path_to_unzip):
    zip_ref = zipfile.ZipFile(path_to_zip, 'r')
    file = zip_ref.extractall(path_to_unzip)
    print(file)
    zip_ref.close()