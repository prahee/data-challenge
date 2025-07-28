# data-challenge

I started by extracting morphological features from grayscale barnacle mask images using OpenCV. This included area, perimeter, circularity, aspect ratio, solidity, and centroid coordinates. I stored the cleaned output in barnacle_features.csv. No missing values, so I didn’t need to impute anything.

Then I explored the feature distributions. The size data was right-skewed, which may sugegst active recruitment across multiple age groups. I used correlation metrics (Pearson and Spearman) to check how traits related to each other. Area and perimeter were tightly linked (r = 0.93), but shape-based traits showed weaker or no correlations with size. I also ran chi-squared tests to double-check if shape and size were statistically independent (they were, p ≈ 0.975).

Next came spatial analysis. I used the centroid coordinates to calculate clustering via nearest-neighbor distances. The barnacles weren’t randomly spread out; they were clustered in tight zones, likely tied to microhabitat preference.

For the modeling side, I assumed a barnacle detection model already exists. I defined what its input/output would look like, then mocked up what a human-in-the-loop correction system might do. The idea was to explore how scientists could interact with a model to validate predictions or flag edge cases.

Lastly, I built a Streamlit app (app.py) for visualizing everything. You can explore the traits, zoom into spatial patterns, and export summaries.
