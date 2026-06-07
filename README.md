# DBSCAN Clustering Explorer

## Run
```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Structure
```
dbscan/
├── app/app.py                       # Streamlit dashboard
├── data/synthetic_clusters.csv      # Synthetic dataset (750×8)
├── notebook/dbscan_analysis.ipynb   # Step-by-step notebook
└── requirements.txt
```

## Hyperparameters
| Parameter | Description |
|---|---|
| **ε (epsilon)** | Neighborhood radius — key parameter |
| **min_samples** | Min neighbors to be a core point |
| **metric** | Distance function (euclidean/manhattan/cosine) |
| **algorithm** | NN search: auto / ball_tree / kd_tree / brute |
| **leaf_size** | Speed/memory tradeoff for tree algorithms |
| **p** | Minkowski power (1=Manhattan, 2=Euclidean) |

## Tips
- Use the **k-NN Distance Plot** tab to find the optimal ε (look for the "elbow")
- DBSCAN labels noise as **−1**
- Does not need number of clusters specified in advance
