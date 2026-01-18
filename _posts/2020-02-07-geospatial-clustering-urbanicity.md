---
layout: single
title: "Measuring Urbanicity Through Hospital Distances"
date: 2020-02-07
categories: [practical-ml]
tags: [r, geospatial, clustering, ggmap, data-visualization]
author_profile: true
toc: true
toc_label: "Contents"
toc_sticky: true
---

How do you measure how "urban" an area is without census data?

Population density seems obvious, but it can be misleading. A suburb with apartment complexes might have higher density than a small downtown. And census boundaries don't always match the reality on the ground.

This project explores a different approach: using hospital locations as a proxy for urbanicity. The hypothesis is simple. Hospitals are expensive to build and maintain. They only exist where there's enough population to support them. So the distance between hospitals tells us something about the population density between them.

---

## The Hypothesis

If two hospitals are 30 miles apart, we can infer there isn't enough population between them to justify building another. If they're 300 meters apart, we're probably in a dense urban core.

By measuring the distances between hospitals and clustering them, we might be able to create a "urbanicity score" that reflects reality better than arbitrary administrative boundaries.

Let's test it.

---

## The Data

The [HIFLD Open Data portal](https://hifld-geoplatform.opendata.arcgis.com/datasets/hospitals) provides a comprehensive dataset of US hospitals with latitude/longitude coordinates, bed counts, and status (open/closed).

```r
library(tidyverse)
library(ggmap)      # For Google Maps integration
library(geosphere)  # For distance calculations
library(sp)         # For spatial data handling

# Load hospital data
Hospitals <- read_csv("Hospitals.csv")

# Quick look: 7,581 hospitals with coordinates
head(Hospitals[, c("NAME", "ADDRESS", "X", "Y", "BEDS", "STATUS")])
```

---

## Building the Distance Matrix

The core of this analysis is a distance matrix: the distance from every hospital to every other hospital. With 7,581 hospitals, that's about 57 million distances.

```r
# Extract coordinates
x <- Hospitals$X  # Longitude
y <- Hospitals$Y  # Latitude

# Create spatial points dataframe
# The projection string tells R we're working with lat/long on a globe
xy <- SpatialPointsDataFrame(
  matrix(c(x, y), ncol = 2),
  data.frame(ID = seq_along(x)),
  proj4string = CRS("+proj=longlat +ellps=WGS84 +datum=WGS84")
)

# Calculate all pairwise distances (in meters)
# distm handles the spherical geometry - it knows the Earth isn't flat
mdist <- distm(xy)

# Label rows and columns with hospital addresses
colnames(mdist) <- Hospitals$ADDRESS
rownames(mdist) <- Hospitals$ADDRESS
```

<details>
<summary><strong>See it with a tiny example</strong></summary>

<p>Imagine 4 hospitals: A, B, C, D. The distance matrix looks like:</p>

|   | A | B | C | D |
|---|---|---|---|---|
| A | 0 | 5km | 12km | 30km |
| B | 5km | 0 | 8km | 25km |
| C | 12km | 8km | 0 | 20km |
| D | 30km | 25km | 20km | 0 |

<p>The diagonal is always 0 (distance from a hospital to itself). The matrix is symmetric (A→B = B→A).</p>

<p>Hospital A's closest neighbor is B (5km). Hospital D is isolated, with its closest neighbor 20km away.</p>

</details>

---

## First Attempt: Minimum Distance

The simplest approach: for each hospital, find the distance to its nearest neighbor.

```r
# Replace 0s with NA (so we don't pick "distance to self" as the minimum)
mdist2 <- mdist
mdist2[mdist2 == 0] <- NA

# Find minimum distance for each hospital
min_distances <- apply(mdist2, 1, min, na.rm = TRUE)

# Create classification based on distance thresholds
# These thresholds are hypothesis-driven, not scientifically validated
Hospitals2 <- Hospitals %>%
  mutate(
    Distance = min_distances,
    Classification = case_when(
      Distance <= 300      ~ "Hospital Center",  # Same campus
      Distance <= 3000     ~ "Very Urban",       # < 3km
      Distance <= 12000    ~ "Urban",            # 3-12km
      Distance <= 20000    ~ "Suburban",         # 12-20km
      Distance <= 30000    ~ "Rural",            # 20-30km
      TRUE                 ~ "Very Rural"        # > 30km
    )
  )
```

**The problem:** This approach has a flaw. Two hospitals might be 100 meters apart (a hospital campus), but there's nothing else around for miles. Using just the minimum distance, we'd label them "Very Urban" when they're actually isolated.

---

## The Fix: Average of K Nearest Distances

Instead of just the closest hospital, we look at the 5 closest and average their distances.

```r
# For each hospital, get the 5 smallest distances
# t(apply(...)) applies the function row-by-row
k <- 5
mdist_sorted <- t(apply(mdist2, 1, sort, na.last = TRUE))[, 1:k]

# Average the k nearest distances
mean_distances <- rowMeans(mdist_sorted, na.rm = TRUE)

# Reclassify using mean distance
Hospitals3 <- Hospitals %>%
  mutate(
    MeanDistance = mean_distances,
    Classification = case_when(
      MeanDistance <= 3000   ~ "Very Urban",
      MeanDistance <= 12000  ~ "Urban",
      MeanDistance <= 20000  ~ "Suburban",
      MeanDistance <= 30000  ~ "Rural",
      TRUE                   ~ "Very Rural"
    )
  ) %>%
  filter(STATUS == "OPEN", BEDS >= 0)
```

**What happens if you change k?** With k=1, you get the original noisy results. Higher k values smooth things out but might miss genuinely dense pockets. k=5 seemed like a reasonable balance for this dataset.

---

## Mapping the Results

Using `ggmap` with Google Maps, we can visualize the classifications. The maps show hospitals as circles, sized by bed count and colored by urbanicity classification.

```r
# Register your Google Maps API key first
# register_google(key = "YOUR_API_KEY")

# Create filtered datasets for each classification
Urban_10 <- filter(Hospitals3, Classification == "Very Urban")
Urban_8 <- filter(Hospitals3, Classification == "Urban")
Suburb_6 <- filter(Hospitals3, Classification == "Suburban")
Rural_4 <- filter(Hospitals3, Classification == "Rural")
Rural_2 <- filter(Hospitals3, Classification == "Very Rural")

# Plot function
plot_urbanicity <- function(location, zoom = 9) {
  basemap <- ggmap(get_googlemap(
    center = location,
    zoom = zoom,
    maptype = "roadmap"
  ))

  basemap +
    geom_point(aes(x = X, y = Y, size = BEDS, fill = Classification),
               pch = 21, data = Urban_10) +
    geom_point(aes(x = X, y = Y, size = BEDS, fill = Classification),
               pch = 21, data = Urban_8) +
    geom_point(aes(x = X, y = Y, size = BEDS, fill = Classification),
               pch = 21, data = Suburb_6) +
    geom_point(aes(x = X, y = Y, size = BEDS, fill = Classification),
               pch = 21, data = Rural_4) +
    geom_point(aes(x = X, y = Y, size = BEDS, fill = Classification),
               pch = 21, data = Rural_2) +
    scale_fill_manual(
      values = c("Very Urban" = "#7b3294",
                 "Urban" = "#c2a5cf",
                 "Suburban" = "#f7f7f7",
                 "Rural" = "#a6dba0",
                 "Very Rural" = "#008837"),
      breaks = c("Very Urban", "Urban", "Suburban", "Rural", "Very Rural")
    ) +
    scale_size(breaks = c(100, 200, 500, 1000), range = c(1, 6)) +
    labs(size = "Beds", fill = "Urbanicity") +
    guides(fill = guide_legend(override.aes = list(size = 5)))
}

# Example: New York City
plot_urbanicity(c(lon = -73.987, lat = 40.745), zoom = 9)
```

---

## Before and After: New York City

The comparison between the single-distance method and the k-nearest method is striking.

**Single Distance Method (k=1):**
- Manhattan shows a chaotic mix of colors
- Parts of Queens appear as "Very Urban" as Manhattan
- No clear gradient from urban core to suburbs

**K-Nearest Method (k=5):**
- Manhattan is uniformly purple (Very Urban)
- Clear gradient: Manhattan → Brooklyn/Queens → Long Island
- Suburban and rural classifications only appear where they make sense

The k-nearest approach captures what we intuitively know: Manhattan is more urban than Staten Island, even if both have hospitals close together.

---

## More Examples

**Pacific Northwest:**
- Seattle shows a clear urban core surrounded by suburbs
- Eastern Washington is correctly classified as rural/very rural
- The gradient from Seattle to the mountains is visible

**Kansas:**
- Kansas City shows as Urban (not Very Urban, which feels right)
- Wichita shows as Urban
- Small towns that appeared urban with k=1 are now correctly classified as Suburban or Rural

---

## Limitations and Extensions

This approach has real limitations:

**What it misses:**
- Specialty hospitals (children's, psychiatric) might cluster differently than general hospitals
- New hospitals are built, old ones close. The data is a snapshot.
- Rural areas with one large regional hospital look "isolated" even if well-served
- The distance thresholds are arbitrary. 3km vs 4km as the urban cutoff is a judgment call.

**Possible improvements:**
- Weight by hospital size (bed count)
- Incorporate drive time instead of straight-line distance
- Use other infrastructure: fire stations, schools, grocery stores
- Validate against census urbanicity measures
- Try different values of k and see how results change

**What this could be useful for:**
- Quick urbanicity estimates for areas without recent census data
- Identifying underserved areas (long distances to hospitals)
- Retail site selection (hospital density correlates with commercial activity)
- Academic research on healthcare access

---

## Summary

1. **Hospital distances can proxy for urbanicity.** Dense hospital clusters indicate dense populations.

2. **Single minimum distance is noisy.** Two hospitals on the same campus can look "very urban" even in rural areas.

3. **K-nearest averaging smooths the noise.** Looking at the 5 nearest hospitals gives a more realistic picture.

4. **The thresholds are hypotheses, not facts.** The 3km/12km/20km/30km boundaries were chosen based on intuition, not validation.

5. **Maps tell the story.** The before/after comparison for NYC shows the improvement clearly.

---

## Full Code

<details>
<summary><strong>Complete R Script</strong></summary>

```r
# Libraries
library(tidyverse)
library(ggmap)
library(geosphere)
library(sp)
library(matrixStats)

# Load data
Hospitals <- read_csv("Hospitals.csv")

# Create spatial points
x <- Hospitals$X
y <- Hospitals$Y
xy <- SpatialPointsDataFrame(
  matrix(c(x, y), ncol = 2),
  data.frame(ID = seq_along(x)),
  proj4string = CRS("+proj=longlat +ellps=WGS84 +datum=WGS84")
)

# Distance matrix
mdist <- distm(xy)
colnames(mdist) <- Hospitals$ADDRESS
rownames(mdist) <- Hospitals$ADDRESS

# Replace 0s with NA
mdist2 <- mdist
mdist2[mdist2 == 0] <- NA

# Get k nearest distances
k <- 5
mdist_sorted <- t(apply(mdist2, 1, sort, na.last = TRUE))[, 1:k]
mean_distances <- rowMeans(mdist_sorted, na.rm = TRUE)

# Classify
Hospitals3 <- Hospitals %>%
  mutate(
    MeanDistance = mean_distances,
    Classification = case_when(
      MeanDistance <= 3000   ~ "Very Urban",
      MeanDistance <= 12000  ~ "Urban",
      MeanDistance <= 20000  ~ "Suburban",
      MeanDistance <= 30000  ~ "Rural",
      TRUE                   ~ "Very Rural"
    )
  ) %>%
  filter(STATUS == "OPEN", BEDS >= 0)

# Split by classification
Urban_10 <- filter(Hospitals3, Classification == "Very Urban")
Urban_8 <- filter(Hospitals3, Classification == "Urban")
Suburb_6 <- filter(Hospitals3, Classification == "Suburban")
Rural_4 <- filter(Hospitals3, Classification == "Rural")
Rural_2 <- filter(Hospitals3, Classification == "Very Rural")

# Plot NYC
# register_google(key = "YOUR_KEY")
nyc_map <- ggmap(get_googlemap(
  center = c(lon = -73.987, lat = 40.745),
  zoom = 9,
  maptype = "roadmap"
))

nyc_map +
  geom_point(aes(x = X, y = Y, size = BEDS, fill = Classification),
             pch = 21, data = Urban_10) +
  geom_point(aes(x = X, y = Y, size = BEDS, fill = Classification),
             pch = 21, data = Urban_8) +
  geom_point(aes(x = X, y = Y, size = BEDS, fill = Classification),
             pch = 21, data = Suburb_6) +
  geom_point(aes(x = X, y = Y, size = BEDS, fill = Classification),
             pch = 21, data = Rural_4) +
  geom_point(aes(x = X, y = Y, size = BEDS, fill = Classification),
             pch = 21, data = Rural_2) +
  scale_fill_manual(
    values = c("Very Urban" = "#7b3294",
               "Urban" = "#c2a5cf",
               "Suburban" = "#f7f7f7",
               "Rural" = "#a6dba0",
               "Very Rural" = "#008837"),
    breaks = c("Very Urban", "Urban", "Suburban", "Rural", "Very Rural")
  ) +
  scale_size(breaks = c(100, 200, 500, 1000), range = c(1, 6)) +
  labs(size = "Beds", fill = "Urbanicity") +
  guides(fill = guide_legend(override.aes = list(size = 5))) +
  theme(legend.position = "right")
```

</details>

---

*This tutorial is a remaster of work I did in early 2020, before I joined my first data science role. I've cleaned up the code and explanations, but kept the core analysis intact.*
