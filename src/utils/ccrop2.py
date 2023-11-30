# --coding:utf-8--
from osgeo import gdal
from argparse import ArgumentParser
import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def ReadDataBy_Rect(filename, bounds):
    '''
    按照指定经纬度坐标范围读取数据
    :param filename:  文件名 ,可以为gdal能读取的任何格式，包括vrt
    :param bounds:  经纬度坐标范围，[xmin,ymin,xmax,ymax]
    :param res: 读取数据的分辨率
    '''
    options = gdal.WarpOptions(outputBoundsSRS='EPSG:4326', outputBounds=bounds, format='VRT')
    ds = gdal.Dataset = gdal.Warp('', filename, options=options)
    geotran = ds.GetGeoTransform()
    proj = ds.GetProjection()
    data = ds.ReadAsArray()
    ds = None
    return geotran, proj, data


if __name__ == "__main__":
    parser = ArgumentParser(description="")
    parser.add_argument("-image_path", type=str, default=r"E:\dataset\tianchi农业作物\train\jingwei_round1_train_20190619\image_1.png")
    parser.add_argument("-label_path", type=str, default=r"E:\dataset\tianchi农业作物\train\jingwei_round1_train_20190619\image_1_label.png")
    parser.add_argument("-save_dir", type=str, default=r"E:\dataset\tianchi农业作物\train\jingwei_round1_train_20190619\crop2")
    arg = parser.parse_args()
    image_path = arg.image_path
    label_path = arg.label_path
    save_image_dir = os.path.join(arg.save_dir, "image")
    save_label_dir = os.path.join(arg.save_dir, "label")
    stride = 512
    target_size = (1024, 1024)

    if not os.path.isdir(save_image_dir): os.makedirs(save_image_dir)
    if not os.path.isdir(save_label_dir): os.makedirs(save_label_dir)
    root_dir, filename = os.path.split(image_path)
    basename, filetype = os.path.splitext(filename)

    image_ds = gdal.Open(image_path)
    image_band1 = image_ds.GetRasterBand(1)
    image_band2 = image_ds.GetRasterBand(2)
    image_band3 = image_ds.GetRasterBand(3)
    # data_type = image_band1.DataType
    image = np.zeros((image_ds.RasterYSize, image_ds.RasterXSize), dtype=np.uint16)
    image = np.expand_dims(image, axis=0)
    image_band1.ReadAsArray(0, 0, image_ds.RasterXSize, image_ds.RasterYSize, buf_obj=image[0,:,:])
    image_band2.ReadAsArray(0, 0, image_ds.RasterXSize, image_ds.RasterYSize, buf_obj=image[1, :, :])
    image_band3.ReadAsArray(0, 0, image_ds.RasterXSize, image_ds.RasterYSize, buf_obj=image[2, :, :])


    if label_path is not None:
        label_ds = gdal.Open(label_path)
        label_band = label_ds.GetRasterBand(1)
        label = np.zeros((label_ds.RasterYSize, label_ds.RasterXSize), dtype=np.uint8)
        label_band.ReadAsArray(0, 0, label_ds.RasterXSize, label_ds.RasterYSize, buf_obj=label)
        label = np.expand_dims(label, axis=0)

    cnt = 0
    csv_pos_list = []

    h, w = image.shape[1], image.shape[2]
    for i in tqdm(range(w // stride - 1)):
        for j in range(h // stride - 1):
            topleft_x = i * stride
            topleft_y = j * stride
            bottomright_x = topleft_x + target_size[0]
            bottomright_y = topleft_y + target_size[1]

            crop_image = image[:, topleft_y:bottomright_y, topleft_x:bottomright_x].copy()
            if label_path is not None:
                crop_label = label[:, topleft_y:bottomright_y, topleft_x:bottomright_x].copy()
            else:
                crop_label = None

            if crop_image.sum() > 0:  # If crop image is not all black
                image_name = os.path.join(save_image_dir, basename + "_" + str(cnt) + ".png")
                driver = gdal.GetDriverByName('GTiff')
                dataset = driver.Create(image_name, crop_image.shape[2], crop_image.shape[1], crop_image.shape[0], data_type)
                for k in range(crop_image.shape[0]):
                    dataset.GetRasterBand(k + 1).WriteArray(crop_image[k])
                dataset.FlushCache()
                dataset = None

                if crop_label is not None:
                    label_name = os.path.join(save_label_dir, basename + "_" + str(cnt) + ".png")
                    driver = gdal.GetDriverByName('GTiff')
                    dataset = driver.Create(label_name, crop_label.shape[2], crop_label.shape[1], crop_label.shape[0], gdal.GDT_Byte)
                    for k in range(crop_label.shape[0]):
                        dataset.GetRasterBand(k + 1).WriteArray(crop_label[k])
                    dataset.FlushCache()
                    dataset = None

                csv_pos_list.append([cnt, topleft_x, topleft_y, bottomright_x, bottomright_y])
                cnt += 1

    df = pd.DataFrame(csv_pos_list, columns=["Index", "TopLeft_X", "TopLeft_Y", "BottomRight_X", "BottomRight_Y"])
    df.to_csv(os.path.join(arg.save_dir, basename + ".csv"), index=False)