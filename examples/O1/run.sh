#!/bin/bash -l


# 1: create the O0 atlas
#brainbuilder atlases -n 6,5,4,3,2,1 -t 700,525,190,353,149,165 -d 10 -o atlas column -a 1000

# 2: create mesh of pia
#python create_mesh.py

# 3: get synthesis parameters and distributions from synthdb database
#synthdb synthesis-inputs pull --species rat --brain-region Isocortex --concatenate

# 4: update cell densities to cell_composition file (constant density for all cell type)
python create_cell_densities.py 10

# 5: place cells
brainbuilder cells place --composition cell_composition_red.yaml \
    --mtype-taxonomy  mtype_taxonomy.tsv \
    --atlas atlas \
    --output nodes.h5

# 6: fix the region name in synthesis and node file (could also use atlas-property in step 5)
python fix_region.py 

# 7: run synthesis
region-grower synthesize-morphologies \
    --input-cells nodes.h5 \
    --tmd-parameters tmd_parameters.json \
    --tmd-distributions tmd_distributions.json \
    --atlas atlas \
    --out-cells nodes_synthesis.h5 \
    --out-morph-dir morphologies \
    --out-morph-ext asc \
    --nb-processes 5 \
    --overwrite \
    --synthesize-axons \
    --region-structure region_structure.yaml


# 8: plot collage
neurocollage -c collage_config.ini

