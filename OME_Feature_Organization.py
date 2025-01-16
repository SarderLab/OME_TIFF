
"""
Updated on Thu Jan 16 2025

Explanation:
This script now fetches each Region's ID directly from XML (Region.attrib['Id'])
and uses that instead of computing OME-based IDs.

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

###############################################################################
# Helper function to read a specific page from OME TIFF
###############################################################################
def read_ome(file_path, page, downsample):
    """
    Reads a specified page from an OME TIFF using tifffile,
    with optional downsampling via slicing.
    """
    with ti.TiffFile(file_path) as tiff:
        data = np.array(
            tiff.series[0].pages[page].asarray()[::downsample, ::downsample]
        )
    print(data, 'data np array')
    return data


###############################################################################
# Script parameters and paths
###############################################################################
segmentations_dir = '/orange/pinaki.sarder/h.lohaan/tiff_output/OME_TIFF/SegmentationDIRs/'
downsample = 1
slide_path = '/orange/pinaki.sarder/h.lohaan/tiff_output/OME_TIFF/Reference/'
excel_sheets = '/orange/pinaki.sarder/h.lohaan/tiff_output/OME_TIFF/ReferenceExcels/'
outdirs = '/orange/pinaki.sarder/h.lohaan/tiff_output/OME_TIFF'

# Minimum polygon areas for certain annotations to filter out small regions
min_size = [30, 30, 24000, 24000, 10, 10]

version = 1

# Object/Annotation metadata
objects = [
    'non-globally-sclerotic glomeruli',
    'globally-sclerotic glomeruli',
    'tubules',
    'arteries-arterioles'
]
contour_list = ['3', '4', '5', '6']
excel_sheet_names = ['Glomeruli', 'Sclerosed glomeruli', 'Tubules', 'Arteries - Arterioles']

column_names = [
    'Object ID',
    'Source file',
    'Mask name',
    'Mask ID',
    'Protocol for mask creation (DOI)',
    'Annotation tool',
    'Object type',
    'x',
    'y',
    'z'
]
ontology_names = [
    'UBERON:0000074',
    'UBERON:0000074',
    'UBERON:0009773',
    'UBERON:0001637'
]
anatomical_structures = [
    'UBERON:0000074',
    'UBERON:0000074',
    'UBERON:0009773',
    'UBERON:0003509'
]
template_names = [
    '/orange/pinaki.sarder/haitham.abdelazim/HuBMAP/Templates/glomeruli-template.xlsx',
    '/orange/pinaki.sarder/haitham.abdelazim/HuBMAP/Templates/glomeruli-template.xlsx',
    '/orange/pinaki.sarder/haitham.abdelazim/HuBMAP/Templates/tubules-template.xlsx',
    '/orange/pinaki.sarder/haitham.abdelazim/HuBMAP/Templates/arteries-arterioles-template.xlsx'
]

doi = 'dx.doi.org/10.17504/protocols.io.dm6gp35p8vzp/v1'
tool = 'FUSION'

file_paths = glob(segmentations_dir + '*.segmentations.ome.tiff')

for file_path in file_paths:
    
    # Prepare output directory for this particular slide
    outdir = os.path.join(
        outdirs,
        file_path.split('/')[-1].split('.segmentations')[0]
    )
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    # Matching XML, WSI, and Excel paths
    xml_path = (
        slide_path
        + file_path.split('/')[-1].split('.segmentations')[0]
        + '.xml'
    )
    wsi_path = (
        slide_path
        + file_path.split('/')[-1].split('.segmentations')[0]
        + '.svs'
    )
    excel_sheet = (
        excel_sheets
        + file_path.split('/')[-1].split('.segmentations')[0]
        + '.xlsx'
    )
    print("Using XML:", xml_path)

    # Update XML with min/max (if your custom function modifies XML or stores bounding boxes)
    write_minmax_to_xml(xml_path)

    # Open the whole-slide image
    slide = openslide.OpenSlide(wsi_path)
    
    # Dictionary for storing region polygons
    # We'll store each region as: {'region_id': <string>, 'polygon': np.array(...) }
    all_contours = {'1': [], '2': [], '3': [], '4': [], '5': [], '6': []}

    # Parse XML and extract region polygons plus region ID
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for Annotation in root.findall("./Annotation"):
        annotationID = Annotation.attrib['Id']
        if annotationID not in ['1', '2', '3', '4', '5', '6']:
            continue
        for Region in Annotation.findall("./*/Region"):
            # Each region has an attribute 'Id' which we can use
            region_id = Region.attrib.get('Id', '0')
            
            verts = []
            for Vert in Region.findall("./Vertices/Vertex"):
                x_coord = int(float(Vert.attrib['X']))
                y_coord = int(float(Vert.attrib['Y']))
                verts.append([x_coord, y_coord])
            
            region_data = {
                'region_id': region_id,
                'polygon': np.array(verts)
            }
            all_contours[annotationID].append(region_data)

    ############################################################################
    # Now process each object type (glomeruli, tubules, arteries, etc.)
    ############################################################################
    for j in range(len(objects)):
        
        # Read a particular page (channel) from the OME TIFF if needed
        # (We keep this step if you still want the segmentation image data,
        # even though we won't use it to get the ID.)
        if j == 0:
            ome_im = read_ome(file_path, j + 2, downsample)
        else:
            ome_im = read_ome(file_path, j + 1, downsample)
        
        # Extract relevant region dictionaries from the all_contours
        contours_regions = all_contours[contour_list[j]]

        # Lists to store final data
        region_ids_xml = []
        centroids_x = []
        centroids_y = []

        # Filter by area if needed, compute centroid
        for region_data in contours_regions:
            region_id = region_data['region_id']
            contours_temp = region_data['polygon']
            
            # Compute contour area
            a = cv2.contourArea(contours_temp)
            if a > min_size[j + 2]:
                # Keep this region
                region_ids_xml.append(region_id)

                # Centroid in downsampled space
                centroid_x = int(np.mean(contours_temp[:, 0]) // downsample)
                centroid_y = int(np.mean(contours_temp[:, 1]) // downsample)
                centroids_x.append(centroid_x)
                centroids_y.append(centroid_y)

        # Read features from the relevant sheet in the Excel file
        sheet_name = excel_sheet_names[j]
        features = pd.read_excel(excel_sheet, sheet_name=sheet_name)

        # For tubules, the script originally took only the first two columns
        if j == 2:
            features = features.iloc[:, 0:2]

        feature_names = list(features.columns)

        # Scale area and radius if relevant
        if 'Area' in features.columns:
            features['Area'] = features['Area'] * (0.25**2)
        if 'Radius' in features.columns:
            features['Radius'] = features['Radius'] * 0.25
        
        # Prepare lines for metadata
        csv_lines = []

        template_df = pd.read_excel(template_names[j], header=None)

        # Choose mask_name for glomeruli vs tubules, etc.
        if j < 2:
            mask_name = 'glomeruli'
        else:
            mask_name = objects[j]

        # Create rows for each region
        for k in range(len(region_ids_xml)):
            current_line = []
            # "Object ID" is now the region_id from XML
            current_line.append(region_ids_xml[k])
            current_line.append(file_path.split('/')[-1])     # Source file
            current_line.append(mask_name)                    # Mask name
            current_line.append(ontology_names[j])            # Mask ID (ontology)
            current_line.append(doi)                          # Protocol DOI
            current_line.append(tool)                         # Annotation tool
            current_line.append(ontology_names[j])            # Object type (ontology again)
            current_line.append(centroids_x[k])               # x centroid
            current_line.append(centroids_y[k])               # y centroid
            current_line.append(0)                            # z (assuming 0)
            csv_lines.append(current_line)

        # Turn the metadata into a DataFrame
        metadata = pd.DataFrame(csv_lines, columns=column_names)

        # Concatenate with the features DataFrame
        # Use ignore_index=False so columns line up by name
        concatenated_df = pd.concat([metadata, features], axis=1, ignore_index=False)
        # Make sure the final columns are the union of column_names + feature_names
        all_cols = column_names + feature_names
        concatenated_df.columns = all_cols

        # If j >= 2, we do the special template concatenation
        if j >= 2:
            # Prepend a row of column names
            column_names_row = pd.DataFrame([concatenated_df.columns], columns=concatenated_df.columns)
            concatenated_df = pd.concat([column_names_row, concatenated_df], ignore_index=True)
            # Re-label columns with integer indices temporarily
            concatenated_df.columns = [x for x in range(len(concatenated_df.columns))]
            # Merge with template
            concatenated_df = pd.concat([template_df, concatenated_df], axis=0, ignore_index=True)
            concatenated_df = concatenated_df.fillna('N/A')

        # Save to Excel
        if j >= 2:
            # For j >= 2, we skip headers because we've manually inserted them
            concatenated_df.to_excel(
                os.path.join(outdir, objects[j] + '-objects.xlsx'),
                index=False, 
                header=False
            )
        else:
            # For glomeruli (non-sclerotic & sclerotic), save with headers
            concatenated_df.to_excel(
                os.path.join(outdir, objects[j] + '-objects.xlsx'),
                index=False
            )

    ###########################################################################
    # Merge non-sclerotic and sclerotic glomeruli (objects[0] and objects[1])
    # Add column "Is Sclerotic" to final glomeruli spreadsheet
    ###########################################################################
    df1_path = os.path.join(outdir, objects[0] + '-objects.xlsx')  # non-sclerotic
    df2_path = os.path.join(outdir, objects[1] + '-objects.xlsx')  # sclerotic

    df1 = pd.read_excel(df1_path, header=None)
    df2 = pd.read_excel(df2_path, header=None)
    df2 = df2.iloc[1:, :]  # remove the first row which is the header row repeated

    # Combine them
    df_all = pd.concat([df1, df2], axis=0, ignore_index=True)

    # Read templates
    template_df = pd.read_excel(template_names[1], header=None)
    template_df2 = pd.read_excel('/orange/pinaki.sarder/haitham.abdelazim/HuBMAP/Templates/glomeruli-combined-template.xlsx', header=None)

    # Insert the template rows on top
    df_all = pd.concat([template_df, df_all], axis=0, ignore_index=True)

    # Add "Is Sclerotic" column
    scler_col = np.concatenate((
        np.zeros((len(df1) - 1, 1)),
        np.ones((len(df2), 1))
    ), axis=0)
    scler_col = pd.DataFrame(scler_col)
    scler_col.columns = ['Is Sclerotic']
    scler_col['Is Sclerotic'][scler_col['Is Sclerotic'] == 1] = 'TRUE'
    scler_col['Is Sclerotic'][scler_col['Is Sclerotic'] == 0] = 'FALSE'

    # Prepend combined template rows to the sclerotic column
    scler_col.columns = [0]
    scler_col = pd.concat([template_df2, scler_col], axis=0, ignore_index=True)

    # Merge the sclerotic column into df_all
    df_all = pd.concat([df_all, scler_col], axis=1)
    df_all = df_all.fillna('N/A')

    # Save final glomeruli file
    df_all.to_excel(
        os.path.join(outdir, 'glomeruli-objects.xlsx'),
        index=False,
        header=False
    )

    # Remove the individual non-sclerotic / sclerotic Excel files
    os.remove(df1_path)
    os.remove(df2_path)
