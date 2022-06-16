import glob
import os
import cv2

if __name__ == '__main__':
    imgs_folder = r"D:\BioLab\img\training_sets\microtubules_comets\img"
    masks_folder = r"D:\BioLab\img\training_sets\microtubules_comets\mask"

    for mask_path in glob.glob(os.path.join(masks_folder, f"*.bmp")):
        found = False
        base_mask_name = os.path.splitext(os.path.basename(mask_path))[0]
        for img_path in glob.glob(os.path.join(imgs_folder, f"*.png")):
            base_img_name = os.path.splitext(os.path.basename(img_path))[0]
            if base_img_name == base_mask_name:
                found = True
                break
        if not found:
            print(f"There is not mask pair for image: {base_mask_name}")




