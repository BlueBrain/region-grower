#!/bin/bash -l
# load venv if needed
. ~/base/bin/activate

# set the mtypes you want to consider
export MTYPES='["L1_DAC","L23_ChC","L4_MC","L5_TPC:A"]'

# 1: create the O0 atlas
brainbuilder atlases -n 6,5,4,3,2,1 -t 700,525,190,353,149,165 -d 10 -o atlas column -a 800

# 2: create mesh of pia
python create_mesh.py

# 3: get synthesis parameters and distributions from synthdb database
#synthdb synthesis-inputs pull --species rat --brain-region Isocortex --concatenate

# 4: update cell densities to cell_composition file (constant density for all cell type)
python create_cell_densities.py 500 # use large enough to get at least 10 (filter in fix_region)

# 5: place cells
brainbuilder cells place --composition cell_composition_red.yaml \
    --mtype-taxonomy  mtype_taxonomy.tsv \
    --atlas atlas \
    --output nodes.h5

# 6: fix the region name in synthesis and node file (could also use atlas-property in step 5)
python fix_region.py 

# 7: run synthesis
rm -rf morphologies
region-grower synthesize-morphologies \
    --input-cells nodes.h5 \
    --tmd-parameters tmd_parameters.json \
    --tmd-distributions tmd_distributions.json \
    --atlas atlas \
    --out-cells nodes_synthesis.h5 \
    --out-morph-dir morphologies \
    --out-morph-ext asc \
    --out-morph-ext h5 \
    --nb-processes 5 \
    --synthesize-axons \
    --region-structure region_structure.yaml


# 8: plot collage
neurocollage -c collage_config.ini --cells-mtypes $MTYPES
python plot_cells.py
python plot_synth_cells.py
