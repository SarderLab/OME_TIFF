#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:33:21 2023

@author: nlucarelli
"""

import numpy as np
import pandas as pd
import tifffile as ti
from xml_to_mask_minmax import get_annotated_ROIs, xml_to_mask, write_minmax_to_xml
import lxml.etree as ET
import tiffslide as openslide
import os
from scipy import stats
import cv2
from matplotlib import path
from skimage.transform import resize
import matplotlib.pyplot as plt
from glob import glob


def read_ome(file_path, page, downsample):
    with ti.TiffFile(file_path) as tiff:
        data = np.array(tiff.series[0].pages[page].asarray()[
                        ::downsample, ::downsample])
        print(data, 'data np array')
    return data


segmentations_dir = '/orange/pinaki.sarder/haitham.abdelazim/HuBMAP/SegmentationDIRs/'

downsample = 1
slide_path = '/orange/pinaki.sarder/haitham.abdelazim/HuBMAP/Reference/'
excel_sheets = '/orange/pinaki.sarder/haitham.abdelazim/HuBMAP/ReferenceExcels/'
outdirs = '/orange/pinaki.sarder/haitham.abdelazim/HuBMAP/output'
min_size = [30, 30, 24000, 24000, 10, 10]

version = 1
objects = ['non-globally-sclerotic glomeruli',
           'globally-sclerotic glomeruli', 'tubules', 'arteries-arterioles']
contour_list = ['3', '4', '5', '6']
excel_sheet_names = ['Glomeruli', 'Sclerosed glomeruli',
                     'Tubules', 'Arteries - Arterioles']
column_names = ['Object ID', 'Source file', 'Mask name', 'Mask ID',
                'Protocol for mask creation (DOI)', 'Annotation tool', 'Object type', 'x', 'y', 'z']
ontology_names = ['UBERON:0000074', 'UBERON:0000074',
                  'UBERON:0009773', 'UBERON:0001637']
anatomical_structures = ['UBERON:0000074',
                         'UBERON:0000074', 'UBERON:0009773', 'UBERON:0003509']
template_names = ['/orange/pinaki.sarder/haitham.abdelazim/HuBMAP/Templates/glomeruli-template.xlsx', '/orange/pinaki.sarder/haitham.abdelazim/HuBMAP/Templates/glomeruli-template.xlsx',
                  '/orange/pinaki.sarder/haitham.abdelazim/HuBMAP/Templates/tubules-template.xlsx', '/orange/pinaki.sarder/haitham.abdelazim/HuBMAP/Templates/arteries-arterioles-template.xlsx']

doi = 'dx.doi.org/10.17504/protocols.io.dm6gp35p8vzp/v1'
tool = 'FUSION'

file_paths = glob(segmentations_dir + '*.segmentations.ome.tiff')

for file_path in file_paths:
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    outdir = os.path.join(outdirs, base_name)

    os.makedirs(outdir, exist_ok=True)

    xml = os.path.join(slide_path, base_name + '.xml')
    nm = os.path.join(slide_path, base_name + '.svs')
    excel_sheet = os.path.join(excel_sheets, base_name + '.xlsx')

    write_minmax_to_xml(xml)

    slide = openslide.OpenSlide(nm)
    wsi = nm

    slideID, slideExt = os.path.splitext(os.path.basename(wsi))
    all_contours = {str(i): [] for i in range(1, 7)}
    tree = ET.parse(xml)
    root = tree.getroot()

    for Annotation in root.findall("./Annotation"):
        annotationID = Annotation.attrib['Id']
        if annotationID in all_contours:
            for Region in Annotation.findall("./*/Region"):
                verts = [[int(float(Vert.attrib['X'])), int(float(Vert.attrib['Y']))] for Vert in Region.findall("./Vertices/Vertex")]
                all_contours[annotationID].append(np.array(verts))

    for j, obj in enumerate(objects):
        ome_im = read_ome(file_path, j + 2 if j == 0 else j + 1, downsample)
        contours_glom = all_contours[contour_list[j]]

        glom_id_xml, glom_id_ome, centroids_x, centroids_y = [], [], [], []

        for i, contours_temp in enumerate(contours_glom):
            if cv2.contourArea(contours_temp) > min_size[j + 2]:
                glom_id_xml.append(i + 1)
                centroid_x = int(np.mean(contours_temp[:, 0]) // downsample)
                centroid_y = int(np.mean(contours_temp[:, 1]) // downsample)

                min_x, max_x = np.min(contours_temp[:, 0]), np.max(contours_temp[:, 0])
                min_y, max_y = np.min(contours_temp[:, 1]), np.max(contours_temp[:, 1])

                ome_crop = ome_im[min_y:max_y + 1, min_x:max_x + 1]

                edge_points = [
                    (contours_temp[np.where(contours_temp[:, 0] == min_x)][0, 0, :], 1, 0),
                    (contours_temp[np.where(contours_temp[:, 0] == max_x)][0, 0, :], -1, 0),
                    (contours_temp[np.where(contours_temp[:, 1] == min_y)][0, 0, :], 0, 1),
                    (contours_temp[np.where(contours_temp[:, 1] == max_y)][0, 0, :], 0, -1)
                ]

                m_ids = [ome_crop[pt[0][1] + pt[2] - min_y, pt[0][0] + pt[1] - min_x] for pt in edge_points]
                id_ome = stats.mode(m_ids).mode[0] if m_ids else 0

                glom_id_ome.append(id_ome)
                centroids_x.append(centroid_x)
                centroids_y.append(centroid_y)

        sheet_name = excel_sheet_names[j]
        features = pd.read_excel(excel_sheet, sheet_name=sheet_name)
        if j == 2:
            features = features.iloc[:, :2]
        features['Area'] *= 0.25 ** 2
        features['Radius'] *= 0.25

        csv_lines = [
            [glom_id_ome[k], file_path.split('/')[-1], 'glomeruli' if j < 2 else objects[j], ontology_names[j], doi, tool, ontology_names[j], centroids_x[k], centroids_y[k], 0]
            for k in range(len(glom_id_ome))
        ]

        metadata = pd.DataFrame(csv_lines, columns=column_names)
        concatenated_df = pd.concat([metadata, features], axis=1, ignore_index=True)
        concatenated_df.columns = column_names + list(features.columns)
        concatenated_df = concatenated_df[concatenated_df['Object ID'] != 0]

        if j >= 2:
            concatenated_df = pd.concat([pd.DataFrame([concatenated_df.columns], columns=concatenated_df.columns), concatenated_df], ignore_index=True)
            concatenated_df.columns = range(len(concatenated_df.columns))
            concatenated_df = pd.concat([pd.read_excel(template_names[j], header=None), concatenated_df], axis=0, ignore_index=True).fillna('N/A')

        concatenated_df.to_excel(os.path.join(outdir, f"{objects[j]}-objects.xlsx"), index=False, header=(j < 2))

    df1 = pd.read_excel(os.path.join(outdir, f"{objects[0]}-objects.xlsx"), header=None)
    df2 = pd.read_excel(os.path.join(outdir, f"{objects[1]}-objects.xlsx"), header=None).iloc[1:, :]

    df_all = pd.concat([df1, df2], axis=0, ignore_index=True)
    df_all = pd.concat([pd.read_excel(template_names[1], header=None), df_all], axis=0, ignore_index=True)

    scler_col = pd.DataFrame(np.concatenate((np.zeros((len(df1) - 1, 1)), np.ones((len(df2), 1))), axis=0), columns=['Is Sclerotic'])
    scler_col['Is Sclerotic'] = scler_col['Is Sclerotic'].map({1: 'TRUE', 0: 'FALSE'})
    scler_col = pd.concat([pd.read_excel('/orange/pinaki.sarder/haitham.abdelazim/HuBMAP/Templates/glomeruli-combined-template.xlsx', header=None), scler_col], axis=0, ignore_index=True)

    df_all = pd.concat([df_all, scler_col], axis=1).fillna('N/A')
    df_all.to_excel(os.path.join(outdir, 'glomeruli-objects.xlsx'), index=False)
    os.remove(os.path.join(outdir, f"{objects[1]}-objects.xlsx"))
