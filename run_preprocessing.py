import  pathconfig
import  os
from preprocessing import create_train_test_validate_dataset
import  argparse
from zipfile import ZipFile


def main(dataset):

    # Collect class names from directory names in './data/UCMerced_LandUse/Images/'
    # opening the zip file in READ mode
    paths = pathconfig.paths()  # object from class pathconfig to extract the path for solutio
    with ZipFile(dataset, 'r') as zip:
        # extracting all the files
        print('Extracting all Dataset UCMerced Land Use from : {}'.format(dataset))
        zip.extractall(path=paths.DATA_DIR)
        print('The Dataset UCMerced Land Use has been extracted inside {}'.format(paths.DATA_DIR))

    sources_dataset = paths.UCMERCED_LANDUSE_DATASET  # the sources path of dataset
    class_names = os.listdir(sources_dataset)  # the class names from dataset
    target_dirs = create_train_test_validate_dataset(sources_dataset)  # the targets dirs for division amoung test trainign and validation

    print('Preprocessing of Dataset UCMerced LandUse:')
    print('Classes : {}'.format(class_names))
    print('Folders created or already present for training/testing/validate : {}'.format(target_dirs))


if __name__ == "__main__":
    paths = pathconfig.paths()  # object from class pathconfig to extract the path for solutio
    parser = argparse.ArgumentParser(description='Python with Multi-modal')
    parser.add_argument('--path_dataset', default=paths.UCMERCED_LANDUSE_DATASET_ZIP, type=str,help='dataset UCMerced Land Use')
    args = parser.parse_args()

    if (args.path_dataset != ""):  #python vgg.py --mode evaluation --dataset test (cmd)
        main(dataset=args.path_dataset)



