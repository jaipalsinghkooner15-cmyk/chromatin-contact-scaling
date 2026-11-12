# Chromatin contact scaling in tumours and cancer cell lines

This is the code behind our paper, *Divergent chromatin contact scaling in primary tumours and cancer cell lines* (Jaipal Singh Kooner and Bingxin Lu, University of Surrey).

## The question

Inside the nucleus, DNA is folded up, so two bits of DNA that are far apart on the chromosome can still end up touching. Hi-C measures how often that happens. If you plot how often two bits touch against how far apart they are, you get a curve that drops as the distance grows. We call the steepness of that drop alpha (α). A steeper drop means DNA rarely makes contacts over long distances.

Most cancer research is done on cell lines grown in the lab. We asked a simple question: do cell lines fold their DNA the same way as real patient tumours?

## What we used

We didn't generate any data ourselves. Everything is public, and in total we used 169 samples from 11 studies.

| Source | What it gave us | Method | Samples |
|---|---|---|---|
| TCGA HiChIP collection (Yost et al. 2025) | patient tumours from 15 cancer types | H3K27ac HiChIP | 69 tumours |
| GSE133928 (Johnstone et al. 2020) | colon tumours, normal colon from the same patients, colon cell lines | Hi-C | 12 tumours, 7 normal tissue, 5 cell lines |
| GSE148079 (Iyyanki et al. 2021) | bladder tumours and cell lines | Hi-C | 5 tumours, 4 cell lines |
| GSE167154 (Kim et al. 2022) | breast tumours, normal breast, breast cell lines | Hi-C | 3 tumours, 1 normal tissue, 6 cell lines |
| GSE253407 (Chang et al. 2025) | a brain tumour (GBM), cell lines and blood cells | Droplet and paired Hi-C | 1 tumour, 2 blood, 6 cell lines |
| GSE229962 (Xie et al. 2024) | GBM stem cell lines grown from patients | Hi-C | 30 cell lines |
| GSE280831 (Breves et al. 2025) | GBM stem cell lines | Hi-C | 4 cell lines |
| GSE226216 (Kim et al. 2023) | liver cancer cell lines | Hi-C | 4 cell lines |
| GSE118629 (Rhie et al. 2019) | prostate cancer cell lines and a normal prostate line | Hi-C | 3 cell lines |
| GSE185069 (Du et al. 2022) | pancreatic cancer cell lines and a normal pancreas line | Hi-C | 3 cell lines |
| GSE92804 (ENCODE) | lung cancer cell line A549 | Hi-C | 4 runs |
| E-MTAB-8851 (Arnould et al. 2021) | DIvA cells with and without DNA breaks, and with cohesin knocked down (Figure 5 only) | Hi-C | 8 files |

We checked every sample against its original record before using it. A few things came out of that:
- The 30 GSE229962 samples looked like tumours in earlier work, but they're stem cell lines grown from patients, so we count them as cell lines.
- We dropped 10 TCGA files that mixed several patients together, one sample that mixed several cell lines, and one sample uploaded twice.

Every decision, with the evidence behind it, is in `data/sample_annotations.csv`.

## What we did

1. **Same method for every sample.** One Python script (`preprocess/02_compute_alpha.py`) reads every file, whatever format it comes in (.hic, .mcool, .pairs, HiC-Pro). It uses the `hictkpy` package to read Hi-C files.
2. **Counting contacts.** Each chromosome is split into 10 kb pieces, and we count how often each pair of pieces touches. We use raw counts with no correction applied, keep only contacts within the same chromosome, and leave out the sex chromosomes.
3. **Building the curve.** For each distance, we divide the number of contacts by the number of piece pairs at that distance. Pairs with no contacts still count; leaving them out was a bug in an earlier version and made α depend on how deeply a sample was sequenced.
4. **Getting α.** We fit a straight line on a log-log plot between 0.1 and 10 Mb. The slope is α.
5. **Each patient counts once.** Repeat runs of the same cells, and cultures from the same patient, are averaged first.
6. **Comparing tumours with cell lines.** We used the Mann-Whitney U test and Welch's t-test, Cohen's d for effect size, and Bonferroni correction when testing several cancer types. Because over half the cell lines are GBM, we also ran a model that allows for cancer type.
7. **Checking the result holds.** We repeated everything using only tumours measured by Hi-C (not HiChIP), and counting related cultures separately. We also thinned deep samples down to tumour sequencing depth, to check depth wasn't behind the result.
8. **DNA breaks and cohesin (Figure 5).** We ran the same method on the DIvA data, fitting α for each chromosome separately, and tested the changes with the Wilcoxon signed-rank test.

All the code is Python, using pandas, numpy, scipy and matplotlib.

## What we found

- Cell lines have a steeper drop than tumours (α of −1.27 against −1.19).
- That holds within cancer types, in all seven types where we had both tumours and cell lines.
- The gap gets bigger, not smaller, when we only use tumours measured by Hi-C.
- Normal cells grown in the lab look like cancer cell lines, while normal tissue looks like tumours. That suggests growing cells in a dish is part of the reason.
- In the DIvA cells, DNA breaks make the drop steeper. Knocking down cohesin, the protein ring that pulls DNA into loops, removes most of that effect.

The paper has the full numbers.

## Where everything is

**In this repository (code and our tables, no raw data):**

```
data/
  sample_annotations.csv   every sample: what it is, its cancer type, and the evidence for its label
  input_files.csv          which file we used for each sample and where to download it
  fig5_inputs.csv          the 8 files used for Figure 5
preprocess/
  02_compute_alpha.py      the main method, used for every sample
  download_and_compute.py  for very large files: downloads one, works out α, deletes it, moves on
  04_build_master_alpha_table.py   puts α and the sample labels together in one table
analysis/
  compare_alpha.py         tumours against cell lines, and all the statistics
  make_figures.py          Figures 1 to 4 and S1 Fig
  fig5_dna_damage.py       Figure 5
  depth_test.py            the sequencing depth check
tests/                     checks that the code works properly
```

Every script starts with a short explanation of what it does and why.

**On our computer (results, too big or not ours to share here):** everything the scripts produce goes into one data folder:

```
scaling_cache/             contact counts for each sample
alpha_results.csv          α for every sample
master_alpha_table.csv     α with the sample labels
stats_results.csv          every statistical test
figures/                   Figures 1 to 4 and S1 Fig, plus figure_numbers.csv
fig5/                      Figure 5 and its numbers
```

## Which script makes which figure

| In the paper | Made by |
|---|---|
| Figures 1, 2, 3, 4 and S1 Fig | `analysis/make_figures.py` |
| Figure 5 | `analysis/fig5_dna_damage.py` |
| The statistics in the text | `analysis/compare_alpha.py` |
| S1 Table | `data/sample_annotations.csv` with the α results |

`make_figures.py` checks its numbers against the statistics before drawing anything, and stops if they don't match. `figures/figure_numbers.csv` lists every number shown on the figures, so the text can be checked against it.

## Running it yourself

You need Python 3.10 or newer. Install everything with:

```
python -m pip install -r requirements.txt
```

Results go into a data folder (by default `OneDrive/Documents/Dissertation`). You can point any script somewhere else with `--data-root`. Run the steps in this order:

```
python preprocess/02_compute_alpha.py --search FOLDER_WITH_THE_DOWNLOADED_FILES
python preprocess/download_and_compute.py --download-dir A_TEMPORARY_FOLDER
python preprocess/04_build_master_alpha_table.py
python analysis/compare_alpha.py
python analysis/make_figures.py
python analysis/fig5_dna_damage.py --search FOLDER_WITH_THE_FIGURE_5_FILES
python analysis/depth_test.py
python -m pytest -q tests
```
