# Notices

## Trained Models and Artifacts (CC0)

The contents of the `output/` directory — ONNX model files, configuration, and
norms — are dedicated to the public domain under
[CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/).

## IPIP Items

The IPIP-BFFM personality items used in this project are from the
[International Personality Item Pool](https://ipip.ori.org/) and are in the
public domain.

## Training Data

Models are trained on the **IPIP-FFM** response dataset published by the
[Open-Source Psychometrics Project](https://openpsychometrics.org/) (OSPP),
specifically the `IPIP-FFM-data-8Nov2018` release
(<https://openpsychometrics.org/_rawdata/IPIP-FFM-data-8Nov2018.zip>).

**Terms.** OSPP publishes its raw datasets for research use at
<https://openpsychometrics.org/_rawdata/>. The responses were collected from
self-selected, anonymous online visitors who, on finishing the test, agreed that
their answers could be saved and used for research. The data are anonymous and
contain no direct identifiers. OSPP does not attach an explicit public-domain
dedication or open license to the raw response data, so we make no public-domain
or free-reuse claim about it and use it solely for research. Because the sample
is self-selected and online, it is not a probability sample of any general
population — see the Limitations sections in `README.md` and
[`docs/research.md`](docs/research.md).

**Quasi-identifiers dropped.** The pipeline ingests only the 50 IPIP item
responses, the five derived domain scores, and `country` (`pipeline/02_load_sqlite.py`,
`select_output_columns`). All other raw metadata fields that could function as
quasi-identifiers are discarded and never reach the SQLite database, the model,
or any published artifact, namely: approximate geolocation
(`lat_appx_lots_of_err`, `long_appx_lots_of_err`); the IP-count field `IPC`
(used only transiently to keep one record per IP, then dropped); response timing
(`introelapse`, `testelapse`, `endelapse`, and the 50 per-item response-latency
columns `<ITEM>_E`, e.g. `EXT1_E`) and the load timestamp (`dateload`); and
screen dimensions (`screenw`, `screenh`).

## References

- Goldberg, L. R. (1999). A broad-bandwidth, public domain, personality
  inventory measuring the lower-level facets of several five-factor models. In
  I. Mervielde, I. Deary, F. De Fruyt, & F. Ostendorf (Eds.), *Personality
  Psychology in Europe* (Vol. 7, pp. 7–28). Tilburg University Press.
- Goldberg, L. R., Johnson, J. A., Eber, H. W., Hogan, R., Ashton, M. C.,
  Cloninger, C. R., & Gough, H. G. (2006). The International Personality Item
  Pool and the future of public-domain personality measures. *Journal of
  Research in Personality, 40*(1), 84–96.
- Donnellan, M. B., Oswald, F. L., Baird, B. M., & Lucas, R. E. (2006). The
  Mini-IPIP scales: Tiny-yet-effective measures of the Big Five factors of
  personality. *Psychological Assessment, 18*(2), 192–203.
- Open-Source Psychometrics Project. (2018). *IPIP-FFM data*
  (`IPIP-FFM-data-8Nov2018`) [Data set].
  <https://openpsychometrics.org/_rawdata/>
