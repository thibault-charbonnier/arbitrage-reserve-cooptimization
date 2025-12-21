# Co-Optimization of Battery Storage Between Energy Arbitrage and Frequency Reserve Markets

Co-optimization framework for Battery Energy Storage Systems between day-ahead energy arbitrage and balancing reserves. Builds a unified optimization model and evaluates revenue/risk using French market data (RTE/Ember).

---

## Quick Start

This repository uses **uv** for dependancies, please use the following commands to run the project :

```bash
# From the repo root

# Create venv if needed
uv venv .venv

# Activate venv
source .venv/Scripts/activate

# Synchronize the dependancies
uv sync
```

---

## Data ingestion & cleaning

This project relies on two public datasets:

- **Day-ahead electricity prices (France)** from **Ember** (hourly time series).
- **Balancing capacity remuneration** (frequency reserves) from **RTE Services** → _Balancing capacity_ view (quarter-hourly remuneration).

Both sources are converted into compact **Parquet** files with harmonized timestamps and units to make downstream optimization reproducible.

---

### 1) Day-ahead electricity prices (energy arbitrage signal)

**Source** Ember – European electricity prices (France), originally provided at an **hourly** resolution.
Link : https://ember-energy.org/data/european-wholesale-electricity-price-data

**Raw structure**

- Country / zone identifiers
- Start/end timestamps
- `Price` in **€/MWh**

**Transformations**

- **Timestamp normalization:** parse timestamps and convert to **UTC**.
- **Resampling to 15 minutes:** the optimization is run on a 15-min grid to match reserve data.  
  Hourly day-ahead prices are therefore expanded to 15-min by **forward-fill**:
  - each hourly price is repeated for its 4 quarter-hours (00:00, 00:15, 00:30, 00:45).
  - this assumes the day-ahead price is **piecewise constant within the hour** (no interpolation to avoid inventing non-tradable prices).

**Final stored format (`data/prices.parquet`)**

- `Datetime` (**UTC**, 15-min grid)
- `price_energy` in **€/MWh**

---

### 2) Balancing capacity remuneration (frequency reserves)

**Source.** RTE Services – **Balancing capacity** dataset (France).
Link : https://www.services-rte.com/fr/telechargez-les-donnees-publiees-par-rte.html?category=market&type=balancing_capacity&subType=procured_reserves
This dataset contains **capacity payments** for being available to provide reserve, not necessarily energy that is actually activated.

**Key interpretation**

- `Price` is expressed in **€/MW/15min**: for a given quarter-hour interval, you are paid this amount per MW of contracted reserve capacity.
- This is **availability remuneration** (capacity payment). Activation may or may not occur and is not modeled in this project.

**Reserve products kept**
We focus on the two standard frequency control products:

- **FCR** (_Frequency Containment Reserve_, “primary reserve”): fast, typically symmetric (UP & DOWN).
- **aFRR** (_automatic Frequency Restoration Reserve_, “secondary reserve”): automatic restoration, usually split into **UP** and **DOWN** products.

**Filtering choices**

- We keep only **`STD`** product types (standard products in the RTE export), which correspond to the standard procurement framework used for **FCR and aFRR**.
- Long-term / non-standard products (e.g., annual items, `SPE0`) are excluded.

**Direction / “Way”**

- `UP` : ability to **increase net injection** (battery: discharge / reduce charge).
- `DOWN` : ability to **decrease net injection** or **increase consumption** (battery: charge / reduce discharge).
- `UP_DOWN` : symmetric capability (ability to provide both directions).

**Timestamp construction**
The raw file provides a `Date` and an `Heures` interval (e.g., `00:00 - 00:15`).  
We convert this into a single timestamp:

- `Datetime` = **start of the quarter-hour interval** (e.g., `00:00 - 00:15` → `00:00`)
- then convert the local timestamp (provided in **UTC+1**) into **UTC**.

So each record at `Datetime = t` represents the remuneration for the **15-min interval starting at t in UTC**.

**Final stored format (`reserves.parquet`).**

- `Datetime` (**UTC**, 15-min grid)
- `Type` in `{FCR, aFRR}`
- `Way` in `{UP, DOWN, UP_DOWN}`
- `price_reserve` in **€/MW/15min**
