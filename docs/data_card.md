# Data Card

## Source and citation

Kevin Hillstrom, *The MineThatData E-Mail Analytics And Data Mining Challenge*, MineThatData blog,
March 2008. <https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html>

| | |
|---|---|
| Primary source | `http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv` (plain http; the server redirects https to http) |
| Fallback mirror | `https://hillstorm1.s3.us-east-2.amazonaws.com/hillstorm_no_indices.csv.gz` (the mirror used by scikit-uplift) |
| Format | CSV with a header row and CRLF line endings; 12 columns, 64,000 rows. The mirror is gzip-compressed, has the same columns in the same order and the same rows in the same order (verified when building this project); it writes zero spend as `0.0` instead of `0`. |
| Integrity | SHA-256 of both files is pinned in `src/abkit/config.py` (`KNOWN_SHA256`). The first download writes `data/raw/manifest.json` (SHA-256, size, source URL, date, whether the fallback was used, any column renames); later runs verify the checksum and skip the download. |
| Loading | `abkit.data.load_to_duckdb()` creates table `raw_customers` in `data/hillstrom.duckdb` with fixed column types (no type inference) and adds `customer_id`. |

## Design

- **Population:** 64,000 customers who purchased within the last twelve months.
- **Randomisation:** each customer was randomly assigned to one of three groups: an email
  featuring men's merchandise, an email featuring women's merchandise, or no email (control).
- **Outcome window:** visits, conversions and spend were tracked for the two weeks after the
  email campaign.
- **Unit of analysis:** the customer. Every row is one customer and one assignment.

## Columns

Types are those of the DuckDB table `raw_customers`. Roles follow project rules 1 and 2: only
pre-treatment columns may be used as features, covariates or segment definitions; `segment` is
the treatment assignment (renamed to `arm` during cleaning); outcomes are never features.

| Column | Type | Role | Meaning and values |
|---|---|---|---|
| `customer_id` | INTEGER | identifier | Added on load: 1-based row number in file order. Not in the source file. |
| `recency` | INTEGER | pre-treatment | Months since last purchase, 1 to 12. |
| `history_segment` | VARCHAR | pre-treatment | Binned past-year spend: `1) $0 - $100`, `2) $100 - $200`, `3) $200 - $350`, `4) $350 - $500`, `5) $500 - $750`, `6) $750 - $1,000`, `7) $1,000 +`. |
| `history` | DOUBLE | pre-treatment | Actual dollar value spent in the past year. |
| `mens` | INTEGER | pre-treatment | 1 if the customer bought men's merchandise in the past year, else 0. |
| `womens` | INTEGER | pre-treatment | 1 if the customer bought women's merchandise in the past year, else 0. |
| `zip_code` | VARCHAR | pre-treatment | Area type: `Urban`, `Surburban` (sic, spelled this way in the source), `Rural`. |
| `newbie` | INTEGER | pre-treatment | 1 if the customer is new in the past twelve months, else 0. |
| `channel` | VARCHAR | pre-treatment | Channels used to purchase in the past year: `Phone`, `Web`, `Multichannel`. |
| `segment` | VARCHAR | assignment | Randomised treatment: `Mens E-Mail`, `Womens E-Mail`, `No E-Mail`. Renamed to `arm` (`mens`, `womens`, `control`) during cleaning. |
| `visit` | INTEGER | outcome | 1 if the customer visited the website in the following two weeks, else 0. |
| `conversion` | INTEGER | outcome | 1 if the customer purchased in the following two weeks, else 0. |
| `spend` | DOUBLE | outcome | Dollars spent in the following two weeks (0 for non-buyers). |

## Known limitations

- **Age.** The experiment ran in 2008; email engagement, merchandising and customer behaviour
  have changed since, so effect sizes should not be transferred to today without care.
- **No cost data.** There is no cost per email, margin or cost of goods, so results are in
  incremental revenue, not profit. Any profit statement needs an assumed margin and is labelled as
  such.
- **No unsubscribe or complaint data.** The downside of emailing a customer (unsubscribes, spam
  complaints, fatigue) is not observed, so "who not to email" can only be judged on observed
  outcomes.
- **No customer IDs.** The file has no identifier; `customer_id` is a row number added on load
  and cannot be linked to anything else.
- **No timestamps.** There are no send, open, visit or purchase times, so time-to-event analysis
  and any real sequential monitoring are impossible; peeking analyses in this project are
  simulations.
- **One two-week window.** Longer-term effects (repeat purchases, retention) are not observed.
- **Coarse pre-treatment data.** Eight pre-treatment variables, several of them binary or binned,
  limit how finely treatment effects can be personalised.

## Terms of use

The source page publishes the data for the challenge but states no licence. This repository
therefore never commits the raw data: it is downloaded at runtime into `data/raw/`, which is
gitignored. The test fixture in `tests/fixtures/hillstrom_fixture.csv` is fully synthetic (see
`tests/fixtures/make_fixture.py`) and contains no real rows.
