#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:33:21 2023

@author: nlucarelli
"""

import numpy as np
import pandas as pd
import tifffile as ti
from xml_to_mask_minmax import get_annotated_ROIs,xml_to_mask,write_minmax_to_xml
import lxml.etree as ET
import tiffslide as openslide
import os
from scipy import stats
import cv2
from matplotlib import path
from skimage.transform import resize
import matplotlib.pyplot as plt
from glob import glob


def read_ome(file_path,page,downsample):
    with ti.TiffFile(file_path) as tiff:

        data = np.array(tiff.series[0].pages[page].asarray()[::downsample,::downsample])
        print(data, 'data np array')
    return data

segmentations_dir = '/orange/pinaki.sarder/haitham.abdelazim/HuBMAP/SegmentationDIRs/'

downsample=1
slide_path = '/orange/pinaki.sarder/haitham.abdelazim/HuBMAP/Reference/'
excel_sheets = '/orange/pinaki.sarder/haitham.abdelazim/HuBMAP/ReferenceExcels/'
outdirs = '/orange/pinaki.sarder/haitham.abdelazim/HuBMAP/output'
min_size = [30,30,24000,24000,10,10]

version=1
objects = ['non-globally-sclerotic glomeruli','globally-sclerotic glomeruli','tubules','arteries-arterioles']
contour_list = ['3','4','5','6']
excel_sheet_names = ['Glomeruli','Sclerosed glomeruli','Tubules','Arteries - Arterioles']
column_names = ['Object ID','Source file','Mask name','Mask Ontology ID','Protocol for mask creation (DOI)','Object type','x','y','z']
ontology_names = ['UBERON:0000074','UBERON:0000074','UBERON:0009773','UBERON:0001637']
anatomical_structures = ['UBERON:0000074','UBERON:0000074','UBERON:0009773','UBERON:0003509']
template_names = ['/orange/pinaki.sarder/nlucarelli/HuBMAP/glomeruli-template.xlsx','/orange/pinaki.sarder/nlucarelli/HuBMAP/glomeruli-template.xlsx','/orange/pinaki.sarder/nlucarelli/HuBMAP/tubules-template.xlsx','/orange/pinaki.sarder/nlucarelli/HuBMAP/arteries-arterioles-template.xlsx']

doi = 'dx.doi.org/10.17504/protocols.io.dm6gp35p8vzp/v1'


file_paths = glob(segmentations_dir + '*.segmentations.ome.tiff')

for file_path in file_paths:

    outdir = outdirs + '/' + file_path.split('/')[-1].split('.segmentations')[0] + '/'

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    #else:
    #    print('Skipping: {}'.format(file_path))
    #    continue

    xml = slide_path + file_path.split('/')[-1].split('.segmentations')[0] + '.xml'
    nm = slide_path + file_path.split('/')[-1].split('.segmentations')[0] + '.svs'
    excel_sheet = excel_sheets + file_path.split('/')[-1].split('.segmentations')[0] + '.xlsx'

    write_minmax_to_xml(xml)

    slide=openslide.OpenSlide(nm)
    wsi=nm

    slideID,slideExt=os.path.splitext(wsi.split('/')[-1])
    all_contours = {'1':[],'2':[],'3':[],'4':[],'5':[],'6':[]}
    # cortex medulla glomeruli scl_glomeruli tubules arteries(ioles)
    tree = ET.parse(xml)
    root = tree.getroot()
    basename=os.path.splitext(xml)[0]
    for Annotation in root.findall("./Annotation"): # for all annotations
        annotationID = Annotation.attrib['Id']
        if annotationID not in ['1','2','3','4','5','6']:
            pass
        else:
            for Region in Annotation.findall("./*/Region"): # iterate on all region
                verts=[]
                for Vert in Region.findall("./Vertices/Vertex"): # iterate on all vertex in region
                    verts.append([int(float(Vert.attrib['X'])),int(float(Vert.attrib['Y']))])
                all_contours[annotationID].append(np.array(verts))


    for j in range(len(objects)):
        if j==0:
            ome_im = read_ome(file_path,j+2,downsample)
        else:
            ome_im = read_ome(file_path,j+1,downsample)

        contours_glom = all_contours[contour_list[j]]


        glom_id_xml = []
        glom_id_ome = []

        centroids_x = []
        centroids_y = []


        for i in range(len(contours_glom)):

            contours_temp = contours_glom[i]
            a=cv2.contourArea(contours_temp)
            if a>min_size[j+2]:
                glom_id_xml.append(i+1)
                centroid_x = int(np.mean(contours_temp[:,0])//downsample)
                centroid_y = int(np.mean(contours_temp[:,1])//downsample)

                min_x = np.min(contours_temp[:,0])
                max_x = np.max(contours_temp[:,0])
                min_y = np.min(contours_temp[:,1])
                max_y = np.max(contours_temp[:,1])

                ome_crop = ome_im[min_y:max_y+1,min_x:max_x+1]

                leftm = contours_temp[np.where(contours_temp[:,0]==min_x),:][0,0,:]
                leftm_id = ome_crop[leftm[1]-min_y,leftm[0]+1-min_x]

                rightm = contours_temp[np.where(contours_temp[:,0]==max_x),:][0,0,:]
                rightm_id = ome_crop[rightm[1]-min_y,rightm[0]-1-min_x]

                upm = contours_temp[np.where(contours_temp[:,1]==min_y),:][0,0,:]
                upm_id = ome_crop[upm[1]+1-min_y,upm[0]-min_x]

                downm = contours_temp[np.where(contours_temp[:,1]==max_y),:][0,0,:]
                downm_id = ome_crop[downm[1]-1-min_y,downm[0]-min_x]
                
                m_ids = [leftm_id,rightm_id,downm_id,upm_id]
                m_ids = [x for x in m_ids if x!=0]
                print(m_ids,'m_ids data')
                
                if len(m_ids) > 0:
                    mode_result = stats.mode(m_ids).mode
                    if isinstance(mode_result, np.ndarray):
                        id_ome = mode_result[0]
                    else:
                        id_ome = mode_result
                else:
                    id_ome = 0


                glom_id_ome.append(id_ome)

                centroids_x.append(centroid_x)
                centroids_y.append(centroid_y)


        sheet_name = excel_sheet_names[j]
        features = pd.read_excel(excel_sheet,sheet_name=sheet_name)
        if j==2:
            features = features.iloc[:, 0:2]
        feature_names = list(features.columns)
        features['Area'] = features['Area']*0.25**2
        features['Radius'] = features['Radius']*0.25

        csv_lines = []

        template_df = pd.read_excel(template_names[j],header=None)

        # template_colnames = template_df.columns


        if j<2:
            mask_name = 'glomeruli'
        else:
            mask_name = objects[j]

        for k in range(len(glom_id_ome)):
            current_line = []
            current_line.append(glom_id_ome[k])
            current_line.append(file_path.split('/')[-1])
            current_line.append(mask_name)
            current_line.append(ontology_names[j])
            current_line.append(doi)
            current_line.append(ontology_names[j])
            current_line.append(centroids_x[k])
            current_line.append(centroids_y[k])
            current_line.append(0)
            csv_lines.append(current_line)

        metadata = pd.DataFrame(csv_lines,columns=column_names)
        concatenated_df = pd.concat([metadata, features], axis=1, ignore_index=True)
        concatenated_df.columns = column_names + feature_names
        concatenated_df = concatenated_df[concatenated_df['Object ID'] != 0]


        if j >= 2:
            column_names_row = pd.DataFrame([concatenated_df.columns], columns=concatenated_df.columns)
            concatenated_df = pd.concat([column_names_row, concatenated_df], ignore_index=True)
            concatenated_df.columns = [x for x in range(len(concatenated_df.columns))]
            concatenated_df = pd.concat([template_df,concatenated_df],axis=0,ignore_index=True)
            concatenated_df = concatenated_df.fillna('N/A')


        #FIGURE OUT HOW TO CONCATENATE BUT KEEP THE HEADERS BELOW IT
        if j >= 2:
            concatenated_df.to_csv(outdir + objects[j] + '-objects.csv',index=False,header=False)
        else:
            concatenated_df.to_csv(outdir + objects[j] + '-objects.csv',index=False)

    df1 = pd.read_csv(outdir + objects[0] + '-objects.csv',header=None)
    df2 = pd.read_csv(outdir + objects[1] + '-objects.csv',header=None)
    df2 = df2.iloc[1:,:]

    df_all = pd.concat([df1,df2],axis=0,ignore_index=True)

    template_df = pd.read_excel(template_names[1],header=None)
    template_df2 = pd.read_excel('/orange/pinaki.sarder/nlucarelli/HuBMAP/glomeruli-combined-template.xlsx',header=None)

    df_all = pd.concat([template_df,df_all],axis=0,ignore_index=True)

    #Make sure this length matches the actual feature length, not the other thing

    scler_col = np.concatenate((np.zeros((len(df1)-1,1)),np.ones((len(df2),1))),axis=0)
    scler_col = pd.DataFrame(scler_col)
    scler_col.columns = ['Is Sclerotic']
    scler_col['Is Sclerotic'][scler_col['Is Sclerotic'] == 1] = 'TRUE'
    scler_col['Is Sclerotic'][scler_col['Is Sclerotic'] == 0] = 'FALSE'

    scler_col.columns = [0]
    scler_col = pd.concat([template_df2,scler_col],axis=0,ignore_index=True)

    df_all = pd.concat([df_all,scler_col],axis=1)
    df_all = df_all.fillna('N/A')
    df_all.to_csv(outdir + 'glomeruli' + '-objects.csv',index=False,header=False)
    os.remove(outdir + objects[0] + '-objects.csv')
    os.remove(outdir + objects[1] + '-objects.csv')
