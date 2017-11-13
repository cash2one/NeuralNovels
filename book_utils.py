import os

path = os.path.expanduser('~/Code/dl/datasets/Gutenberg/')

def get_all_file_names():
    return os.listdir(path)

def get_file_names_written_by(author):
    all_files = get_all_file_names()
    return [file_name for file_name in all_files if file_name.startswith(author)]

def get_file_contents(file_name):
    return open(path + file_name).read()

def get_files_contents(file_names):
    return [get_file_contents(file_name) for file_name in file_names]
