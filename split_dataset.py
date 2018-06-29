import os
import sys
import math
import random
import shutil

TRAIN_RATE = 70
TEST_RATE = 30
OUTPUT_DIR = "splitted-dataset"
TRAINING_DIR = "{}/train".format(OUTPUT_DIR)
TEST_DIR = "{}/test".format(OUTPUT_DIR)


def main():
    if len(sys.argv) < 2:
        print("USAGE: python3 split-dataset.py [DATASET PATH]")
        sys.exit(-1)

    dataset_path = sys.argv[1]

    if not os.path.exists(dataset_path):
        print("Given dataset path does not exist!")
        sys.exit(-2)

    folders = os.listdir(dataset_path)
    print("Total classes: {}".format(len(folders)))
    print("Distribution rate (train/test data): {}/{}".format(TRAIN_RATE, TEST_RATE))
    print("Creating directory '{}'".format(OUTPUT_DIR))
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
        os.mkdir(TRAINING_DIR)
        os.mkdir(TEST_DIR)
        for folder in folders:
            os.mkdir("{}/{}".format(TRAINING_DIR, folder))
            os.mkdir("{}/{}".format(TEST_DIR, folder))
    else:
        print("A folder named '{}' already exists. "
              "Please rename or remove it and restart the script.".format(OUTPUT_DIR))
        sys.exit(-3)

    print("Splitting data...")
    for folder in folders:
        print("Splitting folder '{}'".format(folder))
        entries = os.listdir("{}/{}".format(dataset_path, folder))
        number_of_entries = len(entries)
        training_part_length = math.ceil((number_of_entries / 100) * TRAIN_RATE)
        random.shuffle(entries)
        training_part = entries[:training_part_length]
        test_part = entries[training_part_length:]
        for file in training_part:
            shutil.copyfile("{}/{}/{}".format(dataset_path, folder, file),
                            "{}/{}/{}".format(TRAINING_DIR, folder, file))
        for file in test_part:
            shutil.copyfile("{}/{}/{}".format(dataset_path, folder, file), "{}/{}/{}".format(TEST_DIR, folder, file))

    print("Validating results...")
    test_results()
    print("Done!")


def test_results():
    dataset_path = sys.argv[1]
    folders = os.listdir(dataset_path)
    for folder in folders:
        original_folder = os.listdir("{}/{}".format(dataset_path, folder))
        training_folder = os.listdir("{}/{}".format(TRAINING_DIR, folder))
        test_folder = os.listdir("{}/{}".format(TEST_DIR, folder))
        for entry in original_folder:
            entry_in_training_folder = True if entry in training_folder else False
            entry_in_test_folder = True if entry in test_folder else False
            if (not entry_in_training_folder and not entry_in_test_folder
               or entry_in_training_folder and entry_in_test_folder):
                print("In folder '{}', the file '{}' is either missing or duplicated!".format(folder, entry))


if __name__ == '__main__':
    main()