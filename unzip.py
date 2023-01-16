import os, tarfile, shutil

dir_name = 'C:/Users/jankr/Documents/Jan/Faks/magisterij/Drugi/GaitDatasetB-silh/GaitDatasetB-silh'
extension = ".gz"

os.chdir(dir_name) # change directory from working dir to dir with files

for item in os.listdir(dir_name): # loop through items in dir
    if item.endswith(extension): # check for ".zip" extension
        file_name = os.path.abspath(item) # get full path of files
        print(file_name)
        zip_ref = tarfile.open(file_name) # create zipfile object
        zip_ref.extractall(dir_name) # extract file to dir
        zip_ref.close()  # close file
        save_file_name = file_name[:-7]
        for item2 in os.listdir(save_file_name):
            print(item2)
            if item2[0:2] != 'nm':
                abs_delete = os.path.join(save_file_name, item2)
                shutil.rmtree(abs_delete)

        #os.remove(file_name) # delete zipped file