from xml.etree import ElementTree as ET
import os
import shutil
import tmd

def create_dirs_per_mtype_from_xml(filename='./neuronDB.xml', output_dir='Subtypes/'):
    '''
    Generates directories of mtypes that collect all cells
    of this type, based on the data in filename
    Filename should be of neuronDB.xml type
    '''
    FileDB = ET.parse(filename)
    root = FileDB.findall('listing')[0]
    morphs = root.findall('morphology')

    mtypes = set()
    for m in morphs:
        try:
            # Define mtypes
            mtype = m.find('mtype').text
            # Define suntypes (if they exist)
            if m.find('msubtype').text:
                mtype = mtype + ':' + m.find('msubtype').text
            mtypes.add(mtype)
        except:
            print 'Failed to process', m

    # Create a directory for each m-type, subtype
    for m in mtypes:
        os.mkdir(output_dir + m)

    for m in morphs:
        mtype = output_dir + m.find('mtype').text
        if m.find('msubtype').text:
            mtype = mtype + ':' + m.find('msubtype').text
        # Copy all morphologies to the corresponding directory according to mtypes
        shutil.copy(m.find('name').text + '.h5', mtype + '/' + m.find('name').text + '.h5')


def hack_mtype_problems(output_dir='Subtypes/'):
    '''Generates missing mtypes and removes misclassified L6HPC'''
    # Hack missing morphology m-types
    os.mkdir(output_dir + 'L5_NGC')
    for m in os.listdir(output_dir + 'L6_NGC'):
        shutil.copy(output_dir + 'L6_NGC/' + m, 'L5_NGC/' + m)

    os.mkdir(output_dir + 'L6_BP')
    for m in os.listdir(output_dir + 'L5_BP'):
        shutil.copy(output_dir + 's/L5_BP/' + m, 'L6_BP/' + m)

    # Cleaning up directory of L6_HPC cells to exclude misclassified L6_BPC as L6_HPC
    pop = tmd.io.load_population(output_dir + 'L6_HPC')

    for n in pop.neurons:
        if len(n.apical) > 1:
            print('Too many apical trees for cell: ' + n.name.split('/')[-1] + ' classified as L6_HPC. Cell will be removed for consistency.')
            os.remove(n.name + '.h5')
