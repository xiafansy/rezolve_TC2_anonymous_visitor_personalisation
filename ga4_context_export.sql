-- GA4 cold-start context cells: sessionize + bucket + aggregate IN BigQuery,
-- so the download is a few hundred rows instead of gigabytes of raw events.
--
-- How to run (free, no credit card):
--   1. console.cloud.google.com/bigquery  (any Google account; create a
--      project if prompted -- sandbox mode is enough)
--   2. Paste this whole file into the editor and Run (takes seconds).
--   3. SAVE RESULTS -> CSV (local file) -> save as
--      archive_ga4/ga4_context_cells.csv
--   4. /usr/bin/python3 ga4_prior_calibration.py --cells archive_ga4/ga4_context_cells.csv
--
-- Note: hours are UTC (the store is global) -- read hour_bucket differences
-- as relative, same caveat as the other datasets.

WITH sessions AS (
  SELECT
    user_pseudo_id,
    (SELECT value.int_value FROM UNNEST(event_params)
     WHERE key = 'ga_session_id') AS sid,
    MIN(event_timestamp) AS start_us,
    ANY_VALUE(traffic_source.medium) AS medium,
    ANY_VALUE(traffic_source.source) AS source,
    ANY_VALUE(device.category) AS device,
    ANY_VALUE(geo.continent) AS continent,
    LOGICAL_OR(event_name = 'purchase') AS purchased,
    LOGICAL_OR(event_name = 'add_to_cart') AS carted
  FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
  WHERE (SELECT value.int_value FROM UNNEST(event_params)
         WHERE key = 'ga_session_id') IS NOT NULL
  GROUP BY user_pseudo_id, sid
)
SELECT
  CASE
    WHEN LOWER(IFNULL(medium, '(none)')) IN ('(none)', '(direct)')
         OR LOWER(IFNULL(source, '')) = '(direct)' THEN 'direct'
    WHEN LOWER(medium) = 'organic' THEN 'search'
    WHEN LOWER(medium) IN ('cpc', 'ppc', 'paidsearch', 'display', 'cpm') THEN 'ad'
    WHEN LOWER(medium) = 'email'
         OR LOWER(IFNULL(source, '')) LIKE '%email%' THEN 'email'
    WHEN LOWER(medium) = 'referral' AND REGEXP_CONTAINS(LOWER(IFNULL(source, '')),
         r'instagram|facebook|twitter|t\.co|pinterest|tiktok|youtube|linkedin|reddit')
         THEN 'social'
    WHEN LOWER(medium) = 'referral' THEN 'referral'
    ELSE 'other'
  END AS referrer,
  LOWER(IFNULL(device, 'unknown')) AS device,
  IFNULL(continent, 'unknown') AS continent,
  CASE
    WHEN EXTRACT(HOUR FROM TIMESTAMP_MICROS(start_us)) < 6 THEN 'night'
    WHEN EXTRACT(HOUR FROM TIMESTAMP_MICROS(start_us)) < 12 THEN 'morning'
    WHEN EXTRACT(HOUR FROM TIMESTAMP_MICROS(start_us)) < 18 THEN 'afternoon'
    ELSE 'evening'
  END AS hour_bucket,
  COUNT(*) AS sessions,
  COUNTIF(purchased) AS purchases,
  COUNTIF(carted) AS carts
FROM sessions
GROUP BY referrer, device, continent, hour_bucket
ORDER BY sessions DESC
