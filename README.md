# data-challenge

I started by extracting morphological features from grayscale barnacle mask images using OpenCV. This included area, perimeter, circularity, aspect ratio, solidity, and centroid coordinates. I stored the cleaned output in barnacle_features.csv. No missing values, so I didn’t need to impute anything.

Then I explored the feature distributions. The size data was right-skewed, which may sugegst active recruitment across multiple age groups. I used correlation metrics (Pearson and Spearman) to check how traits related to each other. Area and perimeter were tightly linked (r = 0.93), but shape-based traits showed weaker or no correlations with size. I also ran chi-squared tests to double-check if shape and size were statistically independent (they were, p ≈ 0.975).

Next came spatial analysis. I used the centroid coordinates to calculate clustering via nearest-neighbor distances. The barnacles weren’t randomly spread out; they were clustered in tight zones, likely tied to microhabitat preference.

On the modeling side, I assumed a barnacle detection model already exists. I defined the input and output, and mocked up how scientists might interact with it.The Streamlit app (app.py) lets you interactively explore the data. To run it, make sure you have python installed and then install the core packages. The full analysis pipeline is also in the Jupyter notebook (Barnacle_Data_Exploration.ipynb). You can launch it with jupyter notebook after installing the same dependencies. Everything (images, masks, feature extraction tools, and documentation) is in this repo.

