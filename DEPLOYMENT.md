# Pipeline and Render deployment

## Run the local pipeline

Run these commands from the project root, in order:

```powershell
python -m src.data_ingestion
python -m src.data_transformation
python -m src.model_trainer
```

The stages read `redditDataset.pkl`, create `Dataset.csv`, create
`processed_dataset.csv`, and train/save `sentiment_model.pkl`. The trainer also
logs metrics and the confusion-matrix image to MLflow. These labels and scores
are based on VADER-generated labels, not human-verified sentiment judgments.

Run tests with:

```powershell
python -m unittest discover -s tests -v
```

## Configure automatic Render deployment

1. Push this project to a GitHub repository on the `main` branch.
2. Include the trained `sentiment_model.pkl` in the repository. The Docker build
   intentionally fails if the model artifact is missing. The raw
   `redditDataset.pkl` is not required by the API image.
3. In Render, create a Web Service from the GitHub repository and choose the
   Docker runtime. Configure it to deploy the `main` branch and set the health
   check path to `/`.
4. Create a Render deploy hook for the service.
5. In GitHub, add a repository Actions secret named
   `RENDER_DEPLOY_HOOK_URL` with the full deploy-hook URL as its value.
6. Push to `main`. GitHub Actions runs unit tests, checks that the model is
   present, builds the Docker image, and then calls the Render hook. Render
   rebuilds and deploys the connected repository. Pull requests run tests and
   the Docker build but do not trigger deployment.

The workflow builds the image as a CI verification step; Render builds its own
deployment image from the connected repository when the deploy hook is called.

## Run the API locally

```powershell
python -m uvicorn app:app --reload
```

Visit `http://127.0.0.1:8000/docs` to try the API.
