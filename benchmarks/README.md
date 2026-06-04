# Benchmarks

Benchmark files are **not committed** (too large). Download them separately.

## ISPD 2015 (with region constraints)

Download the 9 circuits with fence regions:
```bash
# Registration not required for ISPD 2015
wget http://www.ispd.cc/contests/15/ispd2015_contest.tar.gz
tar -xzf ispd2015_contest.tar.gz -C ispd2015/
```

Circuits: `mgc_des_perf_a`, `mgc_des_perf_b`, `mgc_edit_dist_a`, `mgc_matrix_mult_b`,
`mgc_matrix_mult_c`, `mgc_pci_bridge32_a`, `mgc_pci_bridge32_b`, `mgc_superblue11_a`,
`mgc_superblue16_a`

## ISPD 2015 (without region constraints, 11 circuits)

Circuits used by DREAMPlace original paper: `des_perf_1`, `fft_1`, `fft_2`, `fft_a`,
`fft_b`, `matrix_mult_1`, `matrix_mult_2`, `matrix_mult_a`, `superblue12`, `superblue14`,
`superblue19`

Available from: http://www.ispd.cc/contests/15/

## ICCAD 2015 (timing-driven)

Requires free academic registration:
- Register at http://iccad-contest.org/2015/
- Download to `iccad2015/`

Circuits: `des_perf_1`, `des_perf_b`, `edit_dist_1`, `fft_2`, `pci_bridge32_a`, `superblue12`

## NanGate 45nm PDK (open source)

```bash
# Available via OpenROAD-flow-scripts
git clone --depth=1 https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts.git /tmp/ofs
cp -r /tmp/ofs/flow/platforms/nangate45 nangate45/
```

## Quick Test (stub mode)

You do not need to download benchmarks to test the framework:
```bash
python evaluator/run_placement.py --benchmark benchmarks/ispd2015_no_region/fft_1 --stub
```

The stub generates synthetic results with the correct format without reading any files.
