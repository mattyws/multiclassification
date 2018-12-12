import gzip
import os
from io import StringIO, BytesIO

def decompressBytesToString(inputBytes):
  """
  decompress the given byte array (which must be valid
  compressed gzip data) and return the decoded text (utf-8).
  """
  bio = BytesIO()
  stream = BytesIO(inputBytes)
  decompressor = gzip.GzipFile(fileobj=stream, mode='r')
  while True:  # until EOF
    chunk = decompressor.read(8192)
    if not chunk:
      decompressor.close()
      bio.seek(0)
      return bio.read().decode("utf-8")
    bio.write(chunk)
  return None

def compressStringToBytes(inputString):
  """
  read the given string, encode it in utf-8,
  compress the data and return it as a byte array.
  """
  bio = BytesIO()
  bio.write(inputString.encode("utf-8"))
  bio.seek(0)
  stream = BytesIO()
  compressor = gzip.GzipFile(fileobj=stream, mode='w')
  while True:  # until EOF
    chunk = bio.read(8192)
    if not chunk:  # EOF?
      compressor.close()
      return stream.getvalue()
    compressor.write(chunk)

mimic_data_path = "/home/mattyws/Documentos/mimic/data/"
compressed_sepsis3_json_files_path  = "/home/mattyws/Documentos/mimic/data/compressed_json_sepsis/"
compressed_nosepsis3_json_files_path  = "/home/mattyws/Documentos/mimic/data/compressed_json_sepsis/no_sepsis/"
sepsis3_json_files_path = mimic_data_path+"json_sepsis/"

#COMPRESS
files_visited = 0
for dir, path, files in os.walk(sepsis3_json_files_path):
    for file in files:
        files_visited += 1
        if files_visited % 100 == 0:
            print("Visited files {}".format(files_visited))
        with open(dir+'/'+file, 'r') as json_file_handler:
            content = json_file_handler.read()
            if "no_sepsis" in dir:
                path_new_file = compressed_nosepsis3_json_files_path+file
            else:
                path_new_file = compressed_sepsis3_json_files_path + file
            with open(path_new_file, 'wb') as compressed_file_handler:
                compressed_file_handler.write(compressStringToBytes(content))