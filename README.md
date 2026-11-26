# 48-hour PM2.5 forecasting in Porto Velho (Rondônia, Brazil)

Every dry season (July to October), smoke from fires fills the air of Porto Velho, in the Brazilian Amazon. This project builds and tests a neural network that forecasts the hourly PM2.5 curve for the next 48 hours at low-cost sensor sites in the city, with an uncertainty band. The network is compared, under rules fixed before the test, with simple baselines and with the CAMS global forecast, which is what public health platforms currently publish for the city.

Final project of the **Future Tech Leaders Brasil — Space Tech Bootcamp 2026**, and groundwork for a master's project. Author: Giullia de Souza Santos ([@giujjia](https://github.com/giujjia)), Computer Science, Federal University of Rondônia (UNIR).

All values below are **PM2.5 measured by the corrected PurpleAir sensors** (Barkjohn et al., 2022 correction), not reference-grade PM2.5. See [Limitations](#limitations).

## Results

The main metric is the mean absolute error (MAE, µg/m³) of the median forecast, averaged over hours +1 to +48, for forecasts issued at local midnight during the dry season, at the two training sensors. Differences are tested with Diebold-Mariano (Harvey-Leybourne-Newbold correction, autocorrelation-robust variance) and Holm adjustment.

| Model | MAE, cross-validation 2021–2023 | MAE, test 2024 |
| :--- | ---: | ---: |
| **Bi-LSTM M1, no future weather (main model)** | **12.0** | **35.9** |
| Bi-LSTM M2 with ERA5 future weather (upper bound) | 11.2 | 38.0 |
| CAMS with 7-day moving bias correction | — | 36.7 |
| Persistence | 15.7 | 37.1 |
| Ridge | 12.4 / 12.1 | 40.3 / 38.1 |
| Persistence of the last 24 h | 15.0 | 41.5 |
| XGBoost | 12.5 / 12.2 | 43.5 / 42.5 |
| Raw CAMS forecast | — | 45.8 |
| Climatology | 20.9 | 55.1 |

Ridge and XGBoost: first value is the version compared with the main model, second with the upper bound. CAMS was not part of cross-validation. 2024 values use the 197 issuance–sensor pairs common to all models.

**What the tests support:**
- In typical years (cross-validation), the network had a lower error than persistence and climatology (p ≤ 0.0001 after Holm).
- In 2024, **the network was not better than persistence nor than CAMS with moving bias correction**: the differences were not significant (with 99 issuances, the test has little power; no significance is not a tie).
- In 2024, both networks had a lower error than the **raw CAMS forecast** and climatology; the main model was also better than Ridge and XGBoost.
- Adding fire detections (version M3) made every model worse in cross-validation, so M3 was not carried to the test.
- Future weather (ERA5) did not improve the network significantly in 2024 (−2.3 µg/m³, 95% CI −5.8 to 1.3).

**Why the error tripled in 2024 (interpretation supported by the data):** the 2024 dry season was far more polluted than the typical training hour. Over the 48 hours after each issuance, measured PM2.5 averaged 87 µg/m³ (maximum 731), while 99% of training hours (2020–2022) were below 116 µg/m³ (maximum 551). The learned models stayed far below the peaks: the median forecast reached at most 218 µg/m³ (main model), 177 (ERA5 upper bound) and 148 (XGBoost), with mean biases of −28, −31 and −38 µg/m³. Persistence and CAMS with moving bias, which follow the recent level, had almost no mean bias (+1.3 and −2.5). The networks' 10–90% uncertainty band covered 64–75% of observations instead of ~80%. The 2024 test therefore measures performance in a year more extreme than the training years.

## Data

| Source | Role | Access |
| :--- | :--- | :--- |
| [PurpleAir](https://www2.purpleair.com/) low-cost sensors (6 sensors at 5 sites: 2 for training; for out-of-sample evaluation, 3 urban sensors at 2 sites, two of them co-located, and 1 rural) | Target and input | Official API, read key |
| [ERA5](https://cds.climate.copernicus.eu/) reanalysis (Copernicus) | Meteorology | Climate Data Store account |
| [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) VIIRS NOAA-20 | Fire detections and fire radiative power | MAP_KEY |
| [CAMS global forecast](https://ads.atmosphere.copernicus.eu/) (Copernicus) | Baseline in the 2024 test | Atmosphere Data Store account and licence |

**The data and the trained weights are not in this repository.** The PurpleAir terms of use (updated 18/07/2022) forbid making the data available in raw form to third parties, limit products built with them to non-public use, and cover AI models trained on them. The scripts in `src/ingest/` download everything again with your own keys. Sources of the full master's plan (GFS, Sentinel-5P, GOES, MERRA-2, CAMS EAC4, INMET, AERONET) have their own scripts in `src/ingest/` but are not used in this delivery.

## How to reproduce

Python 3.14, CPU. The network was trained on a Tesla T4 GPU (Google Colab); on CPU, results differ from the GPU ones by floating-point noise (up to 0.0011 µg/m³ in our check).

```bash
python3 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0
.venv/bin/pip install -r requirements.txt
cp .env.example .env          # PURPLEAIR_READ_KEY and FIRMS_MAP_KEY
.venv/bin/python -m pytest    # 40 tests: availability rules and the 2024 lock
```

The Copernicus key (CDS and ADS) goes in `~/.cdsapirc`, as described on the Copernicus websites. Run everything from the repository root, in this order:

```bash
# 1. Download (each script documents its options in the docstring)
python src/ingest/purpleair.py historico 39077 56831 219595 219981 225327 128603
python src/ingest/era5.py historico 2020 2024
python src/ingest/firms.py historico 2020 2024 1000
python src/ingest/cams.py historico            # CAMS forecast, 2024 dry season

# 2. Target, inputs and input selection (2020–2023 only)
python src/alvo.py
python src/era5_sensores.py
python src/entradas.py
for f in dobra1 dobra2 dobra3 final; do python src/selecao_entradas.py $f; done

# 3. Cross-validation: one model per fold (see the docstring for all options), then evaluation
python src/validacao_cruzada.py bilstm --dobra dobra1 --versao M1 --variante sem --semente 0
python src/avaliar_vc.py

# 4. Save the fold-3 models used in 2024 (checks they reproduce cross-validation exactly)
python src/salvar_modelos_dobra3.py --lote local    # Ridge and XGBoost, CPU
python src/salvar_modelos_dobra3.py --lote colab    # Bi-LSTM, GPU

# 5. Open 2024 (only after pre-registration) and run the test once
python src/alvo.py --ate-ano 2024 --abrir
python src/entradas.py --ate-ano 2024 --abrir
python src/teste_2024.py --ano 2024 --abrir
python src/figuras_resultados.py
```

**The 2024 lock:** every script that touches 2024 refuses to run without `--abrir`, and a test checks this. `python src/teste_2024.py --ensaio` runs the whole test pipeline on 2023 (already seen in cross-validation) to check the code without opening the test year.

## Structure

```
requirements.txt          pinned dependencies
.env.example              API keys (PurpleAir, FIRMS)
src/
  ingest/                 download of each source
  correcao.py             A/B channel quality control and PurpleAir corrections
  alvo.py                 hourly corrected target, end-of-hour labels, valid days
  features.py             availability rules at forecast time and FIRMS filters
  era5_sensores.py        ERA5 per sensor and hour, with derived variables
  entradas.py             sensor × hour table (ERA5, fire power by ring and sector)
  selecao_entradas.py     fixed rule for input selection, per fold
  conjunto.py             windows, versions M1–M3 and decoder variants
  modelos/                bilstm, xgb, ridge, baselines (persistence, climatology)
  validacao_cruzada.py    one model in one fold
  grade_vc.py             Bi-LSTM grid on Colab
  avaliacao.py            main metric, curves, uncertainty, Diebold-Mariano, Holm
  saida_diaria.py         daily alert probabilities (heteroscedastic extended logistic regression)
  avaliar_vc.py           cross-validation evaluation and model selection
  cams_vies_movel.py      raw CAMS and CAMS with moving bias correction
  salvar_modelos_dobra3.py   models used in 2024, checked against cross-validation
  teste_2024.py           2024 test, pre-registered comparisons only
  figuras_resultados.py   result figures
tests/                    automated tests
```

## Method in brief

1. **Target:** hourly PM2.5 from the PurpleAir sensors, with A/B channel quality control, the Barkjohn et al. (2022) correction, end-of-hour labels and valid days (≥ 18 of 24 hours). The target is never interpolated.
2. **Inputs:** only what exists at forecast time. Versions: M1 (past 72 h of PM2.5 and calendar), M2 (M1 + ERA5 meteorology), M3 (M2 + FIRMS fire power). Candidate inputs enter by a rule fixed before looking at their relation with PM2.5.
3. **Model:** encoder-decoder Bi-LSTM with temporal attention; 10%, 50% and 90% quantiles for each of the next 48 hours; pinball loss on log(1 + PM2.5); fixed hyperparameters; mean of 3 seeds. Decoder variants: no future weather (main, usable in practice) and ERA5 "perfect weather" (upper bound, not a forecast).
4. **Selection:** expanding-window temporal cross-validation (evaluated years 2021, 2022, 2023) with a 48 h embargo; the version is chosen within 2% of the best MAE by the lowest Brier score for "day above 50 µg/m³".
5. **Test:** 2024, once, with the fold-3 models, against the pre-registered baselines; daily alerts at 50, 60 and 125 µg/m³ (Brazilian CONAMA thresholds) and sensors never used in training.

## Limitations

Grouped by level of evidence.

**Measured**
- The target is PM2.5 from corrected low-cost sensors. No published collocation of PurpleAir with a reference monitor in the Brazilian Amazon was found. Even in the USA, under smoke, the correction had a normalized RMSE of up to 40% on daily means and up to 55% on hourly means, and only 5 of 15 sites met the EPA targets on hourly means (Barkjohn et al., 2022). Absolute values and alerts inherit this uncertainty.
- Few data: 2 training sensors, with partial coverage in some years; urban sites lie within ~5 km of each other.
- 2024 was far more polluted than the typical training hour, and the uncertainty band was too narrow on polluted days.
- Fire inputs made the models worse in cross-validation; the network started overfitting earlier as inputs grew (inputs per hour: 5 in M1, 23 in M2, 83 in M3; best epoch 6–18 in M1, 2–7 in M2, 1–5 in M3).

**By design**
- The 2024 test is out-of-sample but not blind: daily means of 2024 had been seen in a preliminary check. The confirmatory test is planned for 2026.
- The ERA5 variant uses observed future weather and is an upper bound. A real weather forecast (GFS) is planned for the master's project.
- One correction only (Barkjohn 2022) in this delivery.

**Hypotheses, not tested**
- Why fire inputs did not help: PM2.5 history may already carry the effect of nearby fires; the satellite sees each fire only twice a day; distance to a fire does not tell when its smoke arrives; fixed hyperparameters may be too weak for 83 inputs.
- The correction may err more in Porto Velho (different smoke and very high humidity). Only a collocation can measure it; the FioAres station, operating in Porto Velho since July 2026 (equipment and method still to be confirmed), may allow it.

**Prior work.** PM2.5 in Porto Velho has been characterized with PurpleAir (Jang & Jung, 2023), and CAMS has been evaluated in the southern Amazon (Nassarden et al., 2025). Public PM2.5 forecasts for the city exist (CAMS, via VigiAR and AlertAr Saúde; CPTEC/INPE) but, as far as our search reached, none had been evaluated against local measurements.

## Data attribution and disclaimer

Data from PurpleAir ([purpleair.com](https://www.purpleair.com)). The forecasts of this project are Powered by PurpleAir. Attribution follows the PurpleAir Terms of Service (Section 4.8) and Attribution Guide.

> Underlying air quality data utilized by pm25-forecast-pvh is sourced from PurpleAir, a network of consumer sensors. Sensors are owned and managed by consumers, therefore PurpleAir does not guarantee data accuracy. All insights and claims are solely made by pm25-forecast-pvh.

This is a research project. Its forecasts and alerts do not warrant the accuracy or availability of any information in a health or emergency context and must not be used for health or emergency decisions.

## License

The code is released under the [MIT License](LICENSE). The license covers the code only: the PurpleAir data and the trained weights are not in this repository and remain under the PurpleAir terms of use (see [Data](#data)).

## References

- Barkjohn, K. K. et al. (2022). Correction and accuracy of PurpleAir PM2.5 measurements for extreme wildfire smoke. *Sensors*, 22(24), 9669. https://doi.org/10.3390/s22249669
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
- Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting*, 13(2), 281–291.
- Jang, Y.-W., & Jung, G.-W. (2023). Temporal characteristics and sources of PM2.5 in Porto Velho of Amazon region in Brazil from 2020 to 2022. *Sustainability*, 15(18), 14012. https://doi.org/10.3390/su151814012
- Nassarden, D. et al. (2025). Evaluation and calibration of MERRA-2 and CAMS reanalysis for PM2.5 in a semi-urbanized area in the south of the Amazon. *Air Quality, Atmosphere & Health*, 18(7), 1957–1972. https://doi.org/10.1007/s11869-025-01745-0
