from neurots import NeuronGrower  # noqa: E402 ; pylint: disable=C0413

import json
if __name__ == "__main__":
    p = json.load(open('tmd_parameters.json'))['O0']['L1_DAC']
    d = json.load(open('tmd_distributions.json'))['O0']['L1_DAC']
    grower = NeuronGrower(
        input_parameters=p,
        input_distributions=d,
        # external_diametrizer=external_diametrizer,
        skip_preprocessing=True,
        # context=context,
        # rng_or_seed=rng,
    )
    grower.grow()
    grower.neuron.write("test.asc")
