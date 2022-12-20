import glob
import cv2.cv2 as cv2
import os

if __name__ == '__main__':
    input_folder_img = r"D:\BioLab\img\training_sets\Nucleus_training_img_and_masks\nucleus_training_mathlab_alternative\Anamaris stiched training set\imgs_2048"
    output_folder_img = r"D:\BioLab\img\training_sets\Nucleus_training_img_and_masks\nucleus_training_mathlab_alternative\Anamaris stiched training set\imgs_512"

    input_folder_mask = r"D:\BioLab\img\training_sets\Nucleus_training_img_and_masks\nucleus_training_mathlab_alternative\Anamaris stiched training set\masks_2048"
    output_folder_mask = r"D:\BioLab\img\training_sets\Nucleus_training_img_and_masks\nucleus_training_mathlab_alternative\Anamaris stiched training set\masks_512"

    for img_path in glob.glob(os.path.join(input_folder_img, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        name_1 = os.path.split(img_path)[1].split(".")[0]
        resized_img = cv2.resize(img.astype('uint8'), (512, 512), interpolation=cv2.INTER_AREA)

        output_img_path = os.path.join(output_folder_img, name_1 + ".png")
        cv2.imwrite(output_img_path, resized_img)
        print(output_img_path)

    for img_path in glob.glob(os.path.join(input_folder_mask, "*.bmp")):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        name_1 = os.path.split(img_path)[1].split(".")[0]
        resized_mask = cv2.resize(img.astype('uint8'), (512, 512), interpolation=cv2.INTER_AREA)

        output_mask_path = os.path.join(output_folder_mask, name_1 + ".bmp")
        cv2.imwrite(output_mask_path, resized_mask)
        print(output_mask_path)