import numpy as np
import os
import tiffslide as openslide
import tifffile as ti
import ome_types as ot
from glob import glob

"""
location (tuple) - (x, y) tuple giving the top left pixel in the level 0 reference frame
size (tuple) - (width, height) tuple giving the region size

"""


def run_convert(filepath, resolutions):
    file_ome_tif = filepath

    file_output = filepath + 'f'

    file_ome_xml = filepath.split('tif')[0] + 'xml'

    cmdstring = 'tiffcomment -set ' + file_ome_xml + ' ' + file_ome_tif
    cmdstring2 = 'BF_MAX_MEM=32000M bfconvert -tilex 256 -tiley 256 -pyramid-resolutions ' + \
        str(resolutions)+' -pyramid-scale 2 -compression LZW ' + \
        filepath + ' ' + file_output

    cmdstring3 = 'rm ' + filepath

    os.system(cmdstring)
    os.system(cmdstring2)
    os.system(cmdstring3)

    return file_output


#
# SCRIPT STARTS HERE
slide_dir = '/orange/pinaki.sarder/nlucarelli/Reference/'
save_dir = '/orange/pinaki.sarder/nlucarelli/HuBMAP/'
wsi_ext = ['.svs', '.scn']
slides = []
for ext in wsi_ext:
    slides.extend(glob(slide_dir + '*' + ext))


for slide in slides:
    slide_name = save_dir + slide.split('/')[-1].split('.')[0] + '.ome.tif'

    if os.path.exists(slide_name):
        print('Skipping: {}'.format(slide_name))
        continue

    sl = openslide.OpenSlide(slide)

    # y = sl.properties['tiffslide.bounds-height']
    # x = sl.properties['tiffslide.bounds-width']

    x, y = sl.dimensions

    smallest_dim = x if x < y else y
    resolutions = int(np.floor(np.log2(smallest_dim/256))
                      ) if int(np.floor(np.log2(smallest_dim/256))) < 4 else 4

    img = sl.read_region((0, 0), 0, (x, y))
    img = np.array(img)
    img = img[:, :, :3]
    img = np.transpose(img, axes=(2, 0, 1))
    img = img.astype(np.uint8)

    # tiff_writer = ti.TiffWriter(slide_name,ome=True,bigtiff=True)
    # tiff_writer.write(img,metadata={'axes':'CYX'})
    with ti.TiffWriter(slide_name, bigtiff=True) as tiff:
        tiff.write(
            img,
            photometric='rgb',
            metadata={'SamplesPerPixel': 3}
        )
    # tiff_writer.close()

    tiff_file = ot.from_tiff(slide_name)
    xml_name = slide_name.split('tif')[0]+'xml'
    xml_data = ot.to_xml(tiff_file)
    xml_data = xml_data.replace(
        '<Pixels', '<Pixels PhysicalSizeXUnit="\u03BCm" PhysicalSizeYUnit="\u03BCm"')
    xml_data = xml_data.replace('<Pixels PhysicalSizeXUnit="\u03BCm" PhysicalSizeYUnit="\u03BCm" DimensionOrder="XYCZT"',
                                '<Pixels PhysicalSizeXUnit="\u03BCm" PhysicalSizeYUnit="\u03BCm" DimensionOrder="XYZCT"')
    with open(xml_name, 'wt+') as fh:
        fh.write(xml_data)

    del img

    output_name = run_convert(slide_name, resolutions)
    exit()
