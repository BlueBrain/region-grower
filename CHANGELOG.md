# Changelog

## [1.5.1](https://github.com/BlueBrain/region-grower/compare/1.5.0..1.5.1)

> 7 November 2024

### Chores And Housekeeping

- Fix URLs and metadata and add license headers (Adrien Berchet - [#5](https://github.com/BlueBrain/region-grower/pull/5))

## [1.5.0](https://github.com/BlueBrain/region-grower/compare/1.4.0..1.5.0)

> 7 November 2024

### Fixes

- OS fixing (Alexis Arnaudon - [39d7a9f](https://github.com/BlueBrain/region-grower/commit/39d7a9ff1c364b970d744fabeb27825b1a717506))

### General Changes

- OS (arnaudon - [02cf9c9](https://github.com/BlueBrain/region-grower/commit/02cf9c924e9f6411c40702bac41c5b6c41d9281d))
- Update README.md (Alexis Arnaudon - [df2d2c0](https://github.com/BlueBrain/region-grower/commit/df2d2c0af0207a8facc9af01eece0ffa620076c3))
- Update README.md (Alexis Arnaudon - [70a5730](https://github.com/BlueBrain/region-grower/commit/70a57301feb7c9b2a8b474c21502c71541ca4555))

## [1.4.0](https://github.com/BlueBrain/region-grower/compare/1.3.1..1.4.0)

> 7 November 2024

### Build

- Bump NeuroTS and update the tests (Adrien Berchet - [66caf20](https://github.com/BlueBrain/region-grower/commit/66caf20b4ae48081243c7caa9586089e0d0dd549))

### New Features

- Add support for empty population (Adrien Berchet - [fb0c830](https://github.com/BlueBrain/region-grower/commit/fb0c8305594f1a07868527531fe6dc9216f14533))

## [1.3.1](https://github.com/BlueBrain/region-grower/compare/1.3.0..1.3.1)

> 7 November 2024

### Build

- Freeze docutils version to be compatible with m2r2 (Adrien Berchet - [254da98](https://github.com/BlueBrain/region-grower/commit/254da98fbc54c8e166452d38dd71a63c53530a06))

### Chores And Housekeeping

- Ensure the morphology names are not considered as numbers (Adrien Berchet - [b7af19a](https://github.com/BlueBrain/region-grower/commit/b7af19a50795b6301026947c1246a717f512c808))

### Documentation Changes

- Remove deprecated example (Adrien Berchet - [10b4148](https://github.com/BlueBrain/region-grower/commit/10b4148faf18b95c8b1897a6810365244fbafad2))

## [1.3.0](https://github.com/BlueBrain/region-grower/compare/1.2.9..1.3.0)

> 7 November 2024

### New Features

- Add morphology_producer column (Alexis Arnaudon - [6ec1f98](https://github.com/BlueBrain/region-grower/commit/6ec1f98e34e1d522604d5c5e97692d7d24613a7c))

### Fixes

- Allow for different region names between parameters and region_structure (Alexis Arnaudon - [aad0232](https://github.com/BlueBrain/region-grower/commit/aad0232c29b22646c0e4818a392cd7606d14f1ec))

### Chores And Housekeeping

- Create empty columns before filling them (Adrien Berchet - [c2cab98](https://github.com/BlueBrain/region-grower/commit/c2cab9827ebaa19fd1105720ccbe9c9592ab17cb))
- Configure dask-dataframe to be compatible with pyarrow (Adrien Berchet - [2652ee0](https://github.com/BlueBrain/region-grower/commit/2652ee0014a24297e5c921d5628f86e5af5709b3))

### Changes to Test Assests

- Bump NeuroTS and update the tests (Adrien Berchet - [ef6d3e3](https://github.com/BlueBrain/region-grower/commit/ef6d3e3e917fd49b8b03b4655bdeacc7863565ce))

### CI Improvements

- Fix gitlab config (Adrien Berchet - [f06d3f6](https://github.com/BlueBrain/region-grower/commit/f06d3f6eb29ad995e6efb48ddf5b0fde40b597ea))

## [1.2.9](https://github.com/BlueBrain/region-grower/compare/1.2.8..1.2.9)

> 7 November 2024

### Refactoring and Updates

- Clean RegionMapper (Adrien Berchet - [afc802d](https://github.com/BlueBrain/region-grower/commit/afc802d8137a20b4c929b91d981940caa88b7178))

## [1.2.8](https://github.com/BlueBrain/region-grower/compare/1.2.7..1.2.8)

> 7 November 2024

### Performance Improvements

- Significantly reduce memory consumption for atlas computations (Adrien Berchet - [e50d1b0](https://github.com/BlueBrain/region-grower/commit/e50d1b06dafefdfd53d046a54001430d2b6d5451))

## [1.2.7](https://github.com/BlueBrain/region-grower/compare/1.2.6..1.2.7)

> 7 November 2024

### Performance Improvements

- Dramatically reduce memory consumption when using MPI (Adrien Berchet - [38c50c1](https://github.com/BlueBrain/region-grower/commit/38c50c1dba12898ecf86502f953684ef284eae22))

## [1.2.6](https://github.com/BlueBrain/region-grower/compare/1.2.5..1.2.6)

> 7 November 2024

### New Features

- Add chunksize parameter (Adrien Berchet - [46444cc](https://github.com/BlueBrain/region-grower/commit/46444cc077cd25ea35de2d3e91a712254f1dc07c))

### Performance Improvements

- Remove serialization and simplify the code (Adrien Berchet - [6851a2f](https://github.com/BlueBrain/region-grower/commit/6851a2fffa8b2e20c6e569c6ef498acd7248a94c))

## [1.2.5](https://github.com/BlueBrain/region-grower/compare/1.2.4..1.2.5)

> 7 November 2024

### New Features

- Improve logger for MPI (Adrien Berchet - [71c24fb](https://github.com/BlueBrain/region-grower/commit/71c24fbc4e58a0d9c3ee9509b6bc6816152522af))

## [1.2.4](https://github.com/BlueBrain/region-grower/compare/1.2.3..1.2.4)

> 7 November 2024

### Chores And Housekeeping

- Add logger entries and ensure nb processes &lt;= nb cells (Adrien Berchet - [95e2d00](https://github.com/BlueBrain/region-grower/commit/95e2d00291cb4c42b8c509f5bfda1a1f80666bfa))
- Use importlib instead of pkg_resources which is deprecated (Adrien Berchet - [7c4c5cc](https://github.com/BlueBrain/region-grower/commit/7c4c5cc79adedd5f56550fe6b34cae13b36d27f7))

## [1.2.3](https://github.com/BlueBrain/region-grower/compare/1.2.2..1.2.3)

> 7 November 2024

### New Features

- Add a logger entry for pip freeze (Adrien Berchet - [d0d97b5](https://github.com/BlueBrain/region-grower/commit/d0d97b5c28ead6231dd33d4c4e3063656c93ecf6))

## [1.2.2](https://github.com/BlueBrain/region-grower/compare/1.2.1..1.2.2)

> 7 November 2024

### New Features

- Improve the ways dask can be configured (Adrien Berchet - [18130c0](https://github.com/BlueBrain/region-grower/commit/18130c0420f8be2725fd4b50e431216acc377c41))

## [1.2.1](https://github.com/BlueBrain/region-grower/compare/1.2.0..1.2.1)

> 7 November 2024

### New Features

- Update dask configuration and hide useless logger entries (Adrien Berchet - [50103f3](https://github.com/BlueBrain/region-grower/commit/50103f358e05818b916af699212c2dfc50ed0517))

### CI Improvements

- Increase MPI launch timeout setting (Adrien Berchet - [4dc542e](https://github.com/BlueBrain/region-grower/commit/4dc542e64bea6183f5128c5aecdc433cf4d74cbe))

## [1.2.0](https://github.com/BlueBrain/region-grower/compare/1.1.4..1.2.0)

> 7 November 2024

### New Features

- Extent scaling for basals (Alexis Arnaudon - [93faecd](https://github.com/BlueBrain/region-grower/commit/93faecd205956b1f0f8eba87c413c85c14e62d52))

### Fixes

- Update parameter schema for basal (Alexis Arnaudon - [ce9c6b5](https://github.com/BlueBrain/region-grower/commit/ce9c6b583c6fc1f224a3d566147943722e273e29))

### Chores And Housekeeping

- Remove some warnings (Adrien Berchet - [1267d7a](https://github.com/BlueBrain/region-grower/commit/1267d7ab531a436cbdfb0c02dd7a763a2d19502b))

## [1.1.4](https://github.com/BlueBrain/region-grower/compare/1.1.3..1.1.4)

> 7 November 2024

### New Features

- The dask client is now initialized and closed only in the synthesize() method (Adrien Berchet - [9491081](https://github.com/BlueBrain/region-grower/commit/9491081235b0e28375e818f3cf007e1f93916d5f))

### Fixes

- Shutdown dask at the end of the computation (Adrien Berchet - [2b3175f](https://github.com/BlueBrain/region-grower/commit/2b3175f7e0dfd45bf884543d9330308577fb3883))

### CI Improvements

- Run the py310-MPI job in CI (Adrien Berchet - [6b9e7b6](https://github.com/BlueBrain/region-grower/commit/6b9e7b6b79f7bcf076642a36b67c6c5d32ed0ad9))

## [1.1.3](https://github.com/BlueBrain/region-grower/compare/1.1.2..1.1.3)

> 7 November 2024

### New Features

- Improve validation error messages (Adrien Berchet - [d0063e5](https://github.com/BlueBrain/region-grower/commit/d0063e54b0710032d6bdbcbf25d2ecf06bc31f5f))

## [1.1.2](https://github.com/BlueBrain/region-grower/compare/1.1.1..1.1.2)

> 7 November 2024

### New Features

- Handle no layers (Alexis Arnaudon - [2a0a366](https://github.com/BlueBrain/region-grower/commit/2a0a3662ac454ee828f23d82a7caccec745776a8))

### Fixes

- Update test from NeuroTS changes (Alexis Arnaudon - [7e6443a](https://github.com/BlueBrain/region-grower/commit/7e6443a724fa971d27915e3fa55b35be11f8b7b7))

## [1.1.1](https://github.com/BlueBrain/region-grower/compare/1.1.0..1.1.1)

> 7 November 2024

## [1.1.0](https://github.com/BlueBrain/region-grower/compare/1.0.1..1.1.0)

> 7 November 2024

### New Features

- Propagate mtypes from default region to all others (Adrien Berchet - [5813df4](https://github.com/BlueBrain/region-grower/commit/5813df48d85ce8fb54cb97d3ca8596ce8430dc49))
- Add optional container finalize step (Alexis Arnaudon - [0ecdcae](https://github.com/BlueBrain/region-grower/commit/0ecdcaee8fbe51ef9191fda8557e26dd1af819c5))

### Fixes

- Handle regions with parameters but no layer information (Adrien Berchet - [08aadd8](https://github.com/BlueBrain/region-grower/commit/08aadd8bc7b4703b6a8b68fb1cf4ad6a7a5977d3))

## [1.0.1](https://github.com/BlueBrain/region-grower/compare/1.0.0..1.0.1)

> 7 November 2024

### Build

- Relax constrain Numpy&lt;1.25 (Adrien Berchet - [88d45f2](https://github.com/BlueBrain/region-grower/commit/88d45f246309cba43cee4f61ba2b70bae41f2442))
- Constrain Numpy&lt;1.25 (Adrien Berchet - [319e3f2](https://github.com/BlueBrain/region-grower/commit/319e3f2e65b50e3bc29409d737f5bcf381ee569f))

### New Features

- Handles unknown regions (Alexis Arnaudon - [dca11c6](https://github.com/BlueBrain/region-grower/commit/dca11c69d34f7711b5036e202c950c5efc279f32))

## [1.0.0](https://github.com/BlueBrain/region-grower/compare/0.4.3..1.0.0)

> 7 November 2024

### New Features

- Support several regions at once (Alexis Arnaudon - [76cc9ab](https://github.com/BlueBrain/region-grower/commit/76cc9ab3b7f0894afb7d39e7ffc5e43577edb5b3))

### CI Improvements

- Fix urllib3 requirement for Py38 (Adrien Berchet - [ab6b680](https://github.com/BlueBrain/region-grower/commit/ab6b68066ff150aebd924d21ee20034bb6de9200))

## [0.4.3](https://github.com/BlueBrain/region-grower/compare/0.4.2..0.4.3)

> 7 November 2024

### Chores And Housekeeping

- Apply Copier template (Adrien Berchet - [ab0da83](https://github.com/BlueBrain/region-grower/commit/ab0da83259aad4c3bcc52150d2aad1be46d59bf0))

### CI Improvements

- Setup min_versions job (Adrien Berchet - [ee21db0](https://github.com/BlueBrain/region-grower/commit/ee21db03d2356da199a27bcc99c9c0006b562f17))
- Fix for tox&gt;=4 (Adrien Berchet - [a51546e](https://github.com/BlueBrain/region-grower/commit/a51546e6cf1cfe2b875d99a81bbe3f5913fc8264))
- Increase MPI timeout (Adrien Berchet - [7dd7635](https://github.com/BlueBrain/region-grower/commit/7dd7635f7f12160c94c2fe92416be1aa028fa3e0))

### General Changes

- Update to 3d_angle mode (Alexis Arnaudon - [8621e4e](https://github.com/BlueBrain/region-grower/commit/8621e4eabef694d42a1f21a576fae5cc8846926a))
- Fix f-string (Alexis Arnaudon - [0a21f7d](https://github.com/BlueBrain/region-grower/commit/0a21f7d46b6c0c96e1fe02ee221f987bba8a55ed))
- Fix entry point (Alexis Arnaudon - [42f9a32](https://github.com/BlueBrain/region-grower/commit/42f9a32bb1c22dca006b6acb773fc2dd98328832))

## [0.4.2](https://github.com/BlueBrain/region-grower/compare/0.4.1..0.4.2)

> 7 November 2024

### Refactoring and Updates

- Apply copier template (Adrien Berchet - [855c6dd](https://github.com/BlueBrain/region-grower/commit/855c6dd384354e55c33141c70a8d41a3dcb30a5a))
- Update from Copier template (Adrien Berchet - [61a741c](https://github.com/BlueBrain/region-grower/commit/61a741c4f2965231cabc9dcc9bf35d920ee2d4f0))

### Changes to Test Assests

- Fix cluster creation to ignore missing Bokeh dependency (Adrien Berchet - [9b26479](https://github.com/BlueBrain/region-grower/commit/9b264794a3d618acc3d34b2908c0ceb3b5ac5b37))

### CI Improvements

- Fix coverage for pytest-cov&gt;=4 (Adrien Berchet - [30a8422](https://github.com/BlueBrain/region-grower/commit/30a8422a52188180d3cecc884c8385a5a3d70f39))

## [0.4.1](https://github.com/BlueBrain/region-grower/compare/0.4.0..0.4.1)

> 7 November 2024

### General Changes

- NeuriteType compat with neurom (Alexis Arnaudon - [657d7cf](https://github.com/BlueBrain/region-grower/commit/657d7cfc84370b059c98ac18345bce7e8adda1eb))

## [0.4.0](https://github.com/BlueBrain/region-grower/compare/0.3.1..0.4.0)

> 7 November 2024

### General Changes

- any regions handling (Alexis Arnaudon - [acd8e97](https://github.com/BlueBrain/region-grower/commit/acd8e9705e96905f51c9c7a79973579168b70c22))

## [0.3.1](https://github.com/BlueBrain/region-grower/compare/0.3.0..0.3.1)

> 7 November 2024

### General Changes

- Update for NeuroTS &gt;= 3.1 (Adrien Berchet - [c236c7e](https://github.com/BlueBrain/region-grower/commit/c236c7e894b395073af83c896e72c6acf14602af))
- rename tns into neurots (Alexis Arnaudon - [db1f65b](https://github.com/BlueBrain/region-grower/commit/db1f65b4f04bee96216d6df08ec9a34f601718ac))
- Drop support of py36 and py37 (Adrien Berchet - [c368317](https://github.com/BlueBrain/region-grower/commit/c368317bbde1fd358ced2b09ba73bc49042b7d60))

## [0.3.0](https://github.com/BlueBrain/region-grower/compare/0.2.4..0.3.0)

> 7 November 2024

### General Changes

- Update NeuroM dependency to 3.0.0 (Aleksei Sanin - [4115a3d](https://github.com/BlueBrain/region-grower/commit/4115a3dfd88e07cecae096c4cc7ed2a03cb6cbb1))

## [0.2.4](https://github.com/BlueBrain/region-grower/compare/0.2.3..0.2.4)

> 7 November 2024

### General Changes

- Use pytest template to improve reports and coverage (Adrien Berchet - [e2b6cdb](https://github.com/BlueBrain/region-grower/commit/e2b6cdb9f4e3c275824a6a1858252c14549f3d80))
- Fix changelog to be compatible with the auto-release CI job (Adrien Berchet - [f9f3dda](https://github.com/BlueBrain/region-grower/commit/f9f3dda6a523193dd67092be4100c3b7df07a910))
- Migration from gerrit to GitLab (Adrien Berchet - [acaf67b](https://github.com/BlueBrain/region-grower/commit/acaf67b1025b4b0089e06a1cf3cec38511b39629))
- Fix retry for synthesize (Adrien Berchet - [a83c40a](https://github.com/BlueBrain/region-grower/commit/a83c40a209f854eac8c8d5dbf62241f3c2142061))
- Improve logger entries (Adrien Berchet - [3e9f426](https://github.com/BlueBrain/region-grower/commit/3e9f42614deb299f8e2fa90e27f872bcc20189b3))
- Update NeuroM requirement (Adrien Berchet - [5c54235](https://github.com/BlueBrain/region-grower/commit/5c5423554e3345dda3bb74b341fdaf7ea9e35b84))
- Revert NeuroM requirement to &lt;3 (Adrien Berchet - [9016ee3](https://github.com/BlueBrain/region-grower/commit/9016ee3fda4aee67a3872060c1d2aef6844e6755))
- Improve bb5 constraint (Adrien Berchet - [5379c32](https://github.com/BlueBrain/region-grower/commit/5379c32bdc25a32acf7a03ff2f8b6fc1b430887f))

## [0.2.3](https://github.com/BlueBrain/region-grower/compare/0.2.2..0.2.3)

> 7 November 2024

### General Changes

- fix load filename with . (arnaudon - [c366489](https://github.com/BlueBrain/region-grower/commit/c3664899a60e0a031df51feba51ebfdb7298bcc5))
- update changelog (arnaudon - [5becc11](https://github.com/BlueBrain/region-grower/commit/5becc11f8baea8144be1b5ebd37139a74921e64d))

## [0.2.2](https://github.com/BlueBrain/region-grower/compare/0.2.1..0.2.2)

> 7 November 2024

### General Changes

- Use random number generator and use pickle to save debug infos (Adrien Berchet - [ef54647](https://github.com/BlueBrain/region-grower/commit/ef54647845607fdfa29229f20c09e91c409c009b))
- Add skip_write option to not write synthesized morphologies (arnaudon - [7646c34](https://github.com/BlueBrain/region-grower/commit/7646c34c70915e6757aa89fe1362d0d6f0271e05))
- Fix computation of max_ph in scale_default_barcode (Adrien Berchet - [6ca51ca](https://github.com/BlueBrain/region-grower/commit/6ca51ca8428efc46f06cc875fcca1d4c0284e582))
- fix duplicates in debug_infos (arnaudon - [7a5857e](https://github.com/BlueBrain/region-grower/commit/7a5857e6a6675589de18cf9bce8919f34ee24349))

## [0.2.1](https://github.com/BlueBrain/region-grower/compare/0.2.0..0.2.1)

> 7 November 2024

### General Changes

- small fixx (arnaudon - [2454ba9](https://github.com/BlueBrain/region-grower/commit/2454ba99baddb7e4655626083ec5c8ab5f9ed2c8))
- Revert "Force to update the CHANGELOG before each release" (Alexis Arnaudon - [958eba3](https://github.com/BlueBrain/region-grower/commit/958eba335a250b0721235f5569aff556c2ac8303))
- Force to update the CHANGELOG before each release (Adrien Berchet - [6645e97](https://github.com/BlueBrain/region-grower/commit/6645e977fbcfea8595265c4fc358773b58a67b84))
- update CHANGELOG (arnaudon - [d1f0c75](https://github.com/BlueBrain/region-grower/commit/d1f0c7553be1b5403b02f818bd88b8157fd7748c))
- release 0.2.1 (arnaudon - [d8d2f60](https://github.com/BlueBrain/region-grower/commit/d8d2f60ae493e2ec7b1efb09437dc1d20e1d94f1))

## [0.2.0](https://github.com/BlueBrain/region-grower/compare/0.1.11..0.2.0)

> 7 November 2024

### General Changes

- Add validation of diametrizer parameters, improve doc and fix formatting (Adrien Berchet - [4178013](https://github.com/BlueBrain/region-grower/commit/4178013b832547d42cf8f7551142aa9b88a93629))
- Refactor synthesis cli to use dask (with mpi) (Adrien Berchet - [eb07813](https://github.com/BlueBrain/region-grower/commit/eb078135dc0f1921ec10f3c7a9f20fcf0f060728))
- Import synthesize_morphologies from placement-algorithm (arnaudon - [3bf185f](https://github.com/BlueBrain/region-grower/commit/3bf185fa63cfa5ddc8548ab3aeb8265c4bd4c57f))
- Improve coverage and use pytest (Adrien Berchet - [2e6371e](https://github.com/BlueBrain/region-grower/commit/2e6371ec3901a58abbe0b6aa366c5dcd8ecf2b57))
- Black the code (Adrien Berchet - [3ba33e3](https://github.com/BlueBrain/region-grower/commit/3ba33e32e91e31d1e2943fcb9800b851b7fe2d70))
- Make CLI more consistent and add debug data output (Adrien Berchet - [a400abf](https://github.com/BlueBrain/region-grower/commit/a400abfb6b6b5ba062d05b14aeb819dc8ae73a83))
- depth_min parameter (arnaudon - [13c3417](https://github.com/BlueBrain/region-grower/commit/13c341796d6a85261b3323b20b5f2dac1a50c549))
- Test CI (Adrien Berchet - [b7d3fa4](https://github.com/BlueBrain/region-grower/commit/b7d3fa432cd9dabf4c74a2b6527f3ef001238c57))
- Test CI (Adrien Berchet - [ea01bee](https://github.com/BlueBrain/region-grower/commit/ea01bee3b5f168b75f9a85ea522f3bfe51462ea3))
- L3_TPC:B-&gt; L3_TPC:C (arnaudon - [04e5548](https://github.com/BlueBrain/region-grower/commit/04e5548859cd83346c9b9e0a15b3b88d63898a6c))

## [0.1.11](https://github.com/BlueBrain/region-grower/compare/0.1.10..0.1.11)

> 7 November 2024

### General Changes

- Remove deprecated script [NSETM-730] (Benoît Coste - [6b7721f](https://github.com/BlueBrain/region-grower/commit/6b7721f890cfee4cafa3216c0c0837d5193258ee))
- Use apical sections from TNS instead of apical points (Adrien Berchet - [a136072](https://github.com/BlueBrain/region-grower/commit/a136072dcc5780b18c1465b4a74beb6d8f4efcd3))

## [0.1.10](https://github.com/BlueBrain/region-grower/compare/0.1.9..0.1.10)

> 7 November 2024

### General Changes

- Merge "raise RegionGrowerError for bad scaling" (Benoît Coste - [94bca89](https://github.com/BlueBrain/region-grower/commit/94bca8922be0b70d1e665a2a6a89a3e6b67ff866))
- fix issue in max y extend with orientation (arnaudon - [63d4e0d](https://github.com/BlueBrain/region-grower/commit/63d4e0d79066fd18ede74a1e4b06c2412fc68733))
- Fix atlas mock (Adrien Berchet - [3d14899](https://github.com/BlueBrain/region-grower/commit/3d148995ef9afe4b82e45927eb138804a7888c41))
- raise RegionGrowerError for bad scaling (arnaudon - [1d412c5](https://github.com/BlueBrain/region-grower/commit/1d412c5742959a587280f8cd21ec1c907a44aa4a))

## [0.1.9](https://github.com/BlueBrain/region-grower/compare/0.1.8..0.1.9)

> 7 November 2024

### General Changes

- Add scale logging for validation purpose (Adrien Berchet - [31b62c2](https://github.com/BlueBrain/region-grower/commit/31b62c2cf83518e76fcb9e78765136bcd386265c))

## [0.1.8](https://github.com/BlueBrain/region-grower/compare/0.1.7..0.1.8)

> 7 November 2024

### General Changes

- Refactor of neurite scaling (arnaudon - [54e5988](https://github.com/BlueBrain/region-grower/commit/54e59889c942761fdb0b35d9c777aa94b954b8ea))
- Use brainbuilder in atlas_mock for tests (arnaudon - [d5ca1b9](https://github.com/BlueBrain/region-grower/commit/d5ca1b9e96e8911ccb9a6bb3c14c427a22ec1e7c))
- Add specific parameters validation (Adrien Berchet - [940f67b](https://github.com/BlueBrain/region-grower/commit/940f67bb6688e69837cf1bfe8b7b16dc7ff1b41d))
- Rescale apical points to be consistent with the actual apical neurite points (Adrien Berchet - [4344ed6](https://github.com/BlueBrain/region-grower/commit/4344ed606d0cf3df1088a549c95e56eb7d760ee1))
- Call TNS validator in Context::verify (Benoît Coste - [924d16c](https://github.com/BlueBrain/region-grower/commit/924d16cb5255385568c33a352b175b3d6425559a))

## [0.1.7](https://github.com/BlueBrain/region-grower/compare/0.1.6..0.1.7)

> 7 November 2024

### General Changes

- Set neurite_types for each mtypes independently. (arnaudon - [d856397](https://github.com/BlueBrain/region-grower/commit/d8563973d8586cc5b1ecf40427c5b22ee92b468e))
- Fix bug that prevent TNS validation from passing (Benoît Coste - [03cabac](https://github.com/BlueBrain/region-grower/commit/03cabacb7b645162905e86809c16870d56c03597))
- Bump to version 0.1.7 (Benoît Coste - [db803df](https://github.com/BlueBrain/region-grower/commit/db803dfcf5f2f936bb59a2f768f8677aa211c82b))

## [0.1.6](https://github.com/BlueBrain/region-grower/compare/0.1.5..0.1.6)

> 7 November 2024

### General Changes

- Refactor atlas handling (arnaudon - [c40b36b](https://github.com/BlueBrain/region-grower/commit/c40b36b70b09e9a6fb07fabcd4dc62e632008595))
- simplest scaling from atlas (arnaudon - [94f3182](https://github.com/BlueBrain/region-grower/commit/94f3182ef212d219d78da962e54a2077a05c4fa1))
- Bump v0.1.6 (Benoît Coste - [2a28273](https://github.com/BlueBrain/region-grower/commit/2a282736bebc60ca7653930529e4c872fae4ecc5))

## [0.1.5](https://github.com/BlueBrain/region-grower/compare/0.1.4..0.1.5)

> 7 November 2024

### General Changes

- cli update (.dat, 2 functions, allow for external tmd_parameters) (arnaudon - [81e0fa1](https://github.com/BlueBrain/region-grower/commit/81e0fa1fa7b40c3b8124fa91c7277643c6e2e796))
- updated dependency + fix tests (arnaudon - [5dacd2f](https://github.com/BlueBrain/region-grower/commit/5dacd2f54510a45f117f0cf1acfdddacc6f1e1ae))
- Sphinx does warning as error (Benoît Coste - [98dc360](https://github.com/BlueBrain/region-grower/commit/98dc36051be3b54064a60132903e4040444677b0))
- Bump version to 0.1.5 (Benoît Coste - [d2656f9](https://github.com/BlueBrain/region-grower/commit/d2656f9929d57efe41eff3b38f75091e02d8fc45))

## [0.1.4](https://github.com/BlueBrain/region-grower/compare/0.1.3..0.1.4)

> 7 November 2024

### General Changes

- Return the result of synthesize() in a SynthesisResult object (Benoît Coste - [2898424](https://github.com/BlueBrain/region-grower/commit/2898424d5bfcb0eb0b3719c301d3e1fe20bba3d2))
- Bump version to 0.1.4 (Benoît Coste - [3a51c3d](https://github.com/BlueBrain/region-grower/commit/3a51c3d31990112e0b5a1981b2d28348ebcff210))

## [0.1.3](https://github.com/BlueBrain/region-grower/compare/0.1.2..0.1.3)

> 7 November 2024

### General Changes

- can use external dimetrizer (diameter-synthesis) (Alexis Arnaudon - [5303922](https://github.com/BlueBrain/region-grower/commit/5303922e724f39c97c5fc6badfd816620f26a250))
- tns v2 compatibility (Alexis Arnaudon - [1893032](https://github.com/BlueBrain/region-grower/commit/1893032a2ed875cde0afe45e0aca3216c074832d))
- SpaceContext.synthesize() also return metadata (Benoît Coste - [6a9176e](https://github.com/BlueBrain/region-grower/commit/6a9176eeb8e048741f2c182666301ec6a3262d1a))
- Fix mtype substitution (kanari - [caa0cd4](https://github.com/BlueBrain/region-grower/commit/caa0cd4a863c4a444e43b72c34b96cd6456053da))
- Bump version to v0.1.3 (Benoît Coste - [e02fc69](https://github.com/BlueBrain/region-grower/commit/e02fc6908f72694e2988a6bf0653541e36928278))

## [0.1.2](https://github.com/BlueBrain/region-grower/compare/0.1.1..0.1.2)

> 7 November 2024

### General Changes

- Fix orientation bug caused by shallow copy (Benoît Coste - [5085152](https://github.com/BlueBrain/region-grower/commit/508515248bb0c78ba45efdcb507fea2fffd8bfdd))
- Bump version (Benoît Coste - [35ab1bd](https://github.com/BlueBrain/region-grower/commit/35ab1bd8efac9e987c78ce31cb38f59e185f1ba8))

## [0.1.1](https://github.com/BlueBrain/region-grower/compare/0.1.0..0.1.1)

> 7 November 2024

### General Changes

- Update script to incorporate diameter parameters into distribs (Benoît Coste - [694107a](https://github.com/BlueBrain/region-grower/commit/694107a0536f89087884616d5b0d139971e0a1c1))
- Add 'method' in distrib if missing (Benoît Coste - [ab7ada5](https://github.com/BlueBrain/region-grower/commit/ab7ada55bbde4de92fad0361e458c0975b2d0365))
- Release region-grower==0.1.1 (Arseny V. Povolotsky - [98f593e](https://github.com/BlueBrain/region-grower/commit/98f593e11942863cc4634dc890614df68eaa34c0))

## [0.1.0](https://github.com/BlueBrain/region-grower/compare/0.0.2..0.1.0)

> 7 November 2024

### General Changes

- Add flag to synthesized centered at 0 or not (Benoît Coste - [e5cf79a](https://github.com/BlueBrain/region-grower/commit/e5cf79a92888a09a23542b939681052e2acc5e1e))
- Add diametrization process (Benoît Coste - [b4fdd67](https://github.com/BlueBrain/region-grower/commit/b4fdd671e3fef0c02387e7f4589b6285ea11df19))
- Fix grow in space example (Benoît Coste - [037ca8a](https://github.com/BlueBrain/region-grower/commit/037ca8a81f80835c8064e61408c9cb3859052e00))
- Grow cells accounting for space. (kanari - [b2cb80b](https://github.com/BlueBrain/region-grower/commit/b2cb80bde485354f0f1b73b9aa421da3d0137c8d))
- Experimental script to decompose cells into directories according to mtypes based on xml input (kanari - [2c213b4](https://github.com/BlueBrain/region-grower/commit/2c213b4246719936bd136934c6f4685ec25faf3f))
- Introduced SpaceContext.verify() method (Arseny V. Povolotsky - [53b5896](https://github.com/BlueBrain/region-grower/commit/53b5896134285c47e62aed575eb677eb3e90d731))
- Update generation of distribution files (Benoît Coste - [85f7150](https://github.com/BlueBrain/region-grower/commit/85f7150f60f9a16df4c996f80102bc10b958689d))
- Release region-grower==0.1.0 (Arseny V. Povolotsky - [d33f201](https://github.com/BlueBrain/region-grower/commit/d33f2015179c564ac5ec7446bb8151e6f86c97a5))

## 0.0.2

> 7 November 2024

### General Changes

- Release region-grower==0.0.2 (Arseny V. Povolotsky - [5e486e2](https://github.com/BlueBrain/region-grower/commit/5e486e26eccc474e76ba224f844cee806f092d40))
- Merge "Revised the script for extracting distributions" (Arseniy Povolotskiy - [934c1e2](https://github.com/BlueBrain/region-grower/commit/934c1e2472b14511972fa62fef52bc9ea55c4900))
- Automatic generation of mtype parameters (kanari - [30114ce](https://github.com/BlueBrain/region-grower/commit/30114ce7da9eee96efdfd72b4c72b254f2373a05))
- Fix parameter generation errors (kanari - [a338948](https://github.com/BlueBrain/region-grower/commit/a3389484a173f9e4fc0b4765c57616eb778282d9))
- First commit (Benoît Coste - [ece67b3](https://github.com/BlueBrain/region-grower/commit/ece67b321e4412c05e8694d73bd1ffee47a69772))
- Conform API to circuit building pipeline specification (Benoît Coste - [0058142](https://github.com/BlueBrain/region-grower/commit/00581429aa3779b39a4e0fd894c757282372da83))
- Enable tox (Benoît Coste - [9b17658](https://github.com/BlueBrain/region-grower/commit/9b17658e7b5c2b4a340879dfd997ddb00c513e34))
- Revised the script for extracting distributions (Arseny V. Povolotsky - [e11d831](https://github.com/BlueBrain/region-grower/commit/e11d831eb13f0473316e524ea63c2b604a9ff6b6))
- Add code to extract distributions; includes necessary hacks (kanari - [9ae8100](https://github.com/BlueBrain/region-grower/commit/9ae8100104f6607e3efbfcaef6aef790d3cc0bd4))
- Fixed setup.py (Arseny V. Povolotsky - [a4cc76d](https://github.com/BlueBrain/region-grower/commit/a4cc76df4bc6b5f7356b771ef0c2a36e06d7c90c))
- Adding validation file (Benoît Coste - [2284af0](https://github.com/BlueBrain/region-grower/commit/2284af0524ed08c4aeb8bbb66207b3f7964aeec0))
- Use Atlas and cells instead of circuit config (kanari - [9ca48fe](https://github.com/BlueBrain/region-grower/commit/9ca48fe29a7b9b25a0219f566ccf24c71d796546))
- Fix parameter definition for interneurons (kanari - [1e2758d](https://github.com/BlueBrain/region-grower/commit/1e2758d4c8395f4c75c5126f832b5c7373fe109b))
- Changed SpaceContext construction arguments (Arseny V. Povolotsky - [7b0b9b5](https://github.com/BlueBrain/region-grower/commit/7b0b9b51ccdaac8d87f97e864473827de1f75370))
- Make mayavi dependency optional (kanari - [2f873e3](https://github.com/BlueBrain/region-grower/commit/2f873e367edfad857b2815d2aba23e2d69774b85))
- Release region-grower==0.0.1 (Arseny V. Povolotsky - [062e8c4](https://github.com/BlueBrain/region-grower/commit/062e8c47e7df4918723b39ee86592d09931387c3))
- Remove dependency from bluepy (kanari - [7a090ff](https://github.com/BlueBrain/region-grower/commit/7a090fffa50188056db6270fef3839f9e86275b7))
- Initial empty repository (Dries Verachtert - [3e25b6f](https://github.com/BlueBrain/region-grower/commit/3e25b6fdf88c05e95ac9397128cf99bf2cd069be))
