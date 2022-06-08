import numpy as np
import shutil
from loguru import logger
import os

def SplitterHelper(df, classes, DATASET_PATH, TRAIN_DIR, VALID_DIR, TEST_DIR):
    # Splitting dataset to train and validation
    for label in classes:
        class_df = df.loc[df['class'] == label]
        # 60% train, 20% valid, 20% test
        train, val, test = \
            np.split(class_df.sample(frac=1, random_state=42),
                    [int(.6*len(class_df)), int(.8*len(class_df))])

        for index in range(len(train.index)):
            filename = train.iloc[index]["filename"]
            # source path to image
            src = os.path.join(DATASET_PATH, label, filename)
            # destination path to image
            dst = os.path.join(TRAIN_DIR, label, filename)
            # copy the image from the source to the destination
            shutil.copyfile(src, dst)
        logger.info(
            f"Finish Moving {index+1} Train Image(s) from class {label}")

        for index in range(len(val.index)):
            filename = val.iloc[index]["filename"]
            # source path to image
            src = os.path.join(DATASET_PATH, label, filename)
            # destination path to image
            dst = os.path.join(VALID_DIR, label, filename)
            # copy the image from the source to the destination
            shutil.copyfile(src, dst)
        logger.info(
            f"Finish Moving {index+1} Validation Image(s) from class {label}")

        for index in range(len(test.index)):
            filename = test.iloc[index]["filename"]
            # source path to image
            src = os.path.join(DATASET_PATH, label, filename)
            # destination path to image
            dst = os.path.join(TEST_DIR, label, filename)
            # copy the image from the source to the destination
            shutil.copyfile(src, dst)
        logger.info(
            f"Finish Moving {index+1} Test Image(s) from class {label}")