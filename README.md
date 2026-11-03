# ModelForge Studio
### Build. Train. Test. Understand.

ModelForge Studio is an interactive, browser-based machine-learning workbench built with **Python, Streamlit and scikit-learn**. It brings common decision-tree experimentation tasks into one interface: inspect a dataset, prepare features, train a model, evaluate it, understand its decision logic, make predictions, and export model artefacts.

The repository includes a pre-trained **Weather Prediction** decision-tree model as its first demonstration model. It is an example of how a model can be loaded and explored inside the platform—not a claim that the platform currently supports every machine-learning algorithm or framework.

> **Project status:** Early prototype / active development  
> **Current model family:** scikit-learn Decision Tree Classifier and Decision Tree Regressor  
> **Application:** Streamlit web application

---

## Contents

- [What ModelForge Studio does](#what-modelforge-studio-does)
- [First demonstration model](#first-demonstration-model)
- [Features and platform pages](#features-and-platform-pages)
- [How a user works with the platform](#how-a-user-works-with-the-platform)
- [How model training works](#how-model-training-works)
- [Supported datasets and input data](#supported-datasets-and-input-data)
- [Project structure](#project-structure)
- [Technology stack](#technology-stack)
- [Run the project locally](#run-the-project-locally)
- [Deploy on Streamlit Community Cloud](#deploy-on-streamlit-community-cloud)
- [How the code is organised](#how-the-code-is-organised)
- [Model import and export](#model-import-and-export)
- [Evaluation metrics](#evaluation-metrics)
- [Important limitations](#important-limitations)
- [Security and responsible use](#security-and-responsible-use)
- [Troubleshooting](#troubleshooting)
- [Development and contribution](#development-and-contribution)
- [License](#license)

---

## What ModelForge Studio does

ModelForge Studio is designed for learners, students, developers and data practitioners who want a practical interface for experimenting with tabular datasets and decision-tree models.

A user can:

1. Load the built-in sample weather dataset or upload a CSV file.
2. Inspect rows, columns, missing values and duplicate rows.
3. Choose the value to predict (the target) and the input columns (features).
4. Choose classification or regression, or allow the application to infer a task from the target column.
5. Configure decision-tree training settings.
6. Train a model using a training/test split.
7. Review model scores and possible overfitting.
8. Explore feature importance, tree structure and decision paths.
9. Enter feature values and request a prediction.
10. Import or export a supported decision-tree model and download a model card or report.

The application is currently focused on **tabular supervised learning with decision trees**. The name describes the intended platform direction; it should not be interpreted as support for arbitrary neural networks, deep-learning models, or every algorithm in scikit-learn.

---

## First demonstration model

### Model 001 — Weather Prediction

The repository ships with `model.pkl`, a pre-trained model used to demonstrate the model-loading and prediction experience.

| Property | Details |
|---|---|
| Display name | Weather Prediction |
| Model family | Decision Tree |
| Framework | scikit-learn |
| Artefact | `model.pkl` |
| Role | First pre-trained demonstration model |
| Main use in the app | Load a trained tree, inspect its structure and make predictions |

The application attempts to load `model.pkl` from the project root. It also checks `models/model.pkl` as an alternate location. If neither file is available or the model cannot be loaded, the app creates a fallback demonstration model using generated synthetic weather data.

**Important:** The code currently labels the shipped model's regression target as `Temperature` with unit `°C` when loading it. Confirm the model's original training dataset, exact feature meanings, target definition, training procedure and independent test performance before treating it as a scientifically validated weather forecast model. The fallback dataset is synthetic and is intended for demonstration and experimentation, not operational forecasting.

---

## Features and platform pages

The app presents the following pages in its navigation.

### 1. Weather forecast

A model-specific demonstration page for the bundled weather regression model.

- Presents input controls for the model's features.
- Provides cool-day, typical-day and warm-day presets based on feature statistics.
- Runs the loaded model to estimate its target value.
- Displays temperature bands and general illustrative guidance.
- Shows the decision path taken through the tree.
- Includes a what-if view that varies one numeric input while keeping the other inputs fixed.

This page expects the default model to be a **regressor**. If the loaded default model is a classifier, the page displays a message directing the user to the Prediction Center instead.

### 2. Train studio

The main workspace for training a model from a dataset.

- Starts with the built-in sample weather dataset or accepts a user-uploaded CSV.
- Displays a data preview and basic dataset counts.
- Offers simple cleaning actions:
  - Drop rows with missing values.
  - Fill numeric gaps with the median.
  - Remove duplicate rows.
- Lets the user select a target column and input features.
- Offers automatic task detection, classification and regression choices.
- Provides decision-tree settings such as maximum depth, minimum samples per leaf, minimum samples to split, split criterion and pruning strength.
- Can sweep tree depths and suggest a depth based on held-out test score.
- Trains and activates a model in the current Streamlit session.
- Shows training and test scores and a basic overfitting warning.

**Data handling note:** The training preparation step drops rows with missing values in the selected target and feature columns. The median-fill button is a separate cleaning action for numeric columns; it is not an automatic imputation pipeline embedded in the trained model.

### 3. Model analytics

The analytics page helps the user inspect a model's structure and evaluate it.

- Shows tree depth, number of leaves and leaf-size information.
- Estimates overfitting risk from tree structure.
- Accepts an evaluation CSV containing the model's feature columns and a true target column.
- Calculates task-appropriate evaluation metrics.
- Displays actual-versus-predicted and residual plots for regression.
- Displays a confusion matrix and classification metrics for classification.
- Offers five-fold cross-validation when the model's training data is available in the current session.

For an imported model, the original training dataset is not included in the pickle by this application. Consequently, cross-validation may not be available. The user can provide a separate labelled evaluation CSV to score the imported model.

### 4. Tree explorer

A visual explanation page for the active decision tree.

- Shows tree depth, leaf count and median leaf size.
- Ranks feature importance.
- Displays a tree diagram with a selectable number of levels.
- Plots the distribution of training records across leaves.
- Shows a text version of the tree rules at a selected depth.
- Displays a structural overfitting-risk indicator.

Feature importance is the tree model's impurity-based importance. It is useful for exploring the fitted tree, but it is not proof of causation and can be misleading when features are correlated or have different measurement characteristics.

### 5. Prediction center

A general prediction interface for the active supported decision-tree model.

- Creates an input control for each model feature.
- Accepts values for a single prediction.
- Displays a predicted class or numeric value.
- For classification, displays class probabilities when available.
- Shows the sequence of tree decisions leading to the prediction.
- Warns when numeric inputs fall outside ranges inferred from the model's split thresholds.

The range warning is an indicator, not a guarantee that a prediction is valid or invalid.

### 6. Model hub

The model exchange area for the active model.

- Downloads the active decision tree as a `.pkl` file.
- Downloads a JSON model card containing model metadata, feature information, hyperparameters and available metrics.
- Accepts `.pkl` and `.joblib` model files for import.
- Checks that the loaded object is a supported scikit-learn decision-tree classifier or regressor.
- Displays the scikit-learn version used by the running app.

Only import pickle/joblib files from sources you trust. Loading these formats can execute arbitrary code.

### 7. Data explorer

A visual exploration page for the current working dataset.

- Correlation heatmap for numeric columns.
- Histograms and box plots for selected columns.
- Two-dimensional scatter plots with optional colour grouping.
- Optional linear trend line for uncoloured numeric relationships.
- Three-dimensional scatter plot when at least three numeric columns are available.
- Missing-value counts and visualisation.

The current working dataset is the sample dataset or the CSV selected in Train Studio. The explorer does not automatically perform advanced statistical inference or data validation.

### 8. Report

Creates a downloadable Markdown report for the active model.

The report can include:

- Model source, task and target.
- Input features and number of training rows.
- Tree depth, leaves and median leaf size.
- Structural overfitting-risk estimate.
- Evaluation metrics, if evaluation data is available.
- Feature-importance values.
- Model hyperparameters.

---

## How a user works with the platform

A typical workflow is:

### A. Start with the pre-trained model

1. Launch ModelForge Studio.
2. Keep **Pre-trained model** selected in the sidebar.
3. Open Model Hub to review the active model and download its model card.
4. Open Prediction Center to enter feature values and run a prediction.
5. Open Tree Explorer to inspect how the model makes decisions.
6. Open Model Analytics to review its structure or provide labelled evaluation data.

### B. Train a model using your own CSV

1. Open **Train studio**.
2. Select **Upload CSV** and upload a dataset.
3. Inspect the preview, missing-value count and duplicate-row count.
4. Apply the available cleaning actions if needed.
5. Select the **target** column—the value the model should learn to predict.
6. Select the **input features**—the columns the model will use to make its prediction.
7. Choose Auto-detect, Regression or Classification.
8. Configure the tree settings.
9. Optionally select **Suggest a depth** and review the training/test score chart.
10. Select **Train model**.
11. The newly trained model becomes the active **My trained model** model for the current session.

### C. Inspect and use the trained model

1. Open Model Analytics to review scores and possible overfitting.
2. Open Tree Explorer to inspect feature importance and tree rules.
3. Open Prediction Center to test an individual input.
4. Open Model Hub to export the model and its JSON model card.
5. Open Report to download a Markdown summary.

### D. Evaluate an imported model

1. Import a supported `.pkl` or `.joblib` decision tree in Model Hub.
2. Activate the imported model.
3. Open Model Analytics.
4. Upload a labelled CSV with all required feature columns and the true target column.
5. Select the true target column and run evaluation.

---

## How model training works

The current training workflow is implemented in `core.py`.

### 1. Select columns

The user selects one target column and one or more feature columns. The target is removed from the input feature list if it is also selected there.

### 2. Prepare rows and features

The current preparation code:

- Selects the target and chosen feature columns.
- Drops rows containing missing values in those selected columns.
- Converts boolean feature columns to integers.
- Treats non-numeric feature columns as categorical.
- Converts categorical values to strings.
- One-hot encodes categorical feature columns using pandas.
- Records numeric feature ranges and categorical values for later input handling.

The resulting encoded feature matrix is used to fit the decision tree.

### 3. Determine the task

The application can infer the task from the target column:

- Non-numeric or boolean targets are treated as classification.
- Numeric targets containing integer-like values and no more than 10 unique values are treated as classification.
- Other numeric targets are treated as regression.

The user can override automatic detection in Train Studio. Regression requires a numeric target. Classification targets are label-encoded for model training.

Automatic detection is a heuristic. Review the selected task before training, especially when a numeric target represents category codes.

### 4. Split the data

The code uses `train_test_split` with a fixed random seed (`42`) and a user-selected test percentage. For classification, it attempts to stratify the split when every class has at least two examples.

At least 30 complete rows are required by the current preparation function.

### 5. Fit the decision tree

The platform fits one of:

- `DecisionTreeClassifier` for classification.
- `DecisionTreeRegressor` for regression.

The model is created with a fixed random seed and the selected tree parameters.

### 6. Review performance

The training page reports:

- Training accuracy for classification or training R² for regression.
- Held-out test accuracy for classification or held-out test R² for regression.
- Tree depth and leaf counts.
- A simple warning when the training score is substantially higher than the test score.

These scores are useful diagnostics, but a single train/test split does not establish that a model will generalise to every real-world dataset.

---

## Supported datasets and input data

### Dataset file type

- CSV (`.csv`) for training and evaluation uploads.

### Practical dataset expectations

For the current workflow:

- Include a header row with column names.
- Include one target column.
- Include at least one usable feature column.
- Provide at least 30 complete rows after selected-column missing rows are removed.
- Use numeric targets for regression.
- Ensure the evaluation CSV contains the feature columns expected by the model plus a column with the true target values.
- Keep feature names and meanings consistent between training, evaluation and prediction.

### Categorical columns

Non-numeric feature columns are converted to categorical string values and one-hot encoded during training. The application stores the observed category values in the active model bundle so it can build the corresponding encoded input columns for predictions in that session.

The current import format accepts a raw scikit-learn decision tree and infers its feature names from `feature_names_in_` when available. It does not restore the full preprocessing history of an arbitrary external pipeline. For reliable reuse, imported models should be compatible with the app's expected feature representation.

### Large datasets

The app loads uploaded CSV data into memory using pandas. Very large files may require substantial memory and may be slow to process. There is no current streaming or distributed-training workflow.

---

## Project structure

```text
modelforge-studio/
│
├── app.py                  # Streamlit interface, pages, navigation and session workflow
├── core.py                 # Dataset preparation, training, evaluation and tree utilities
├── theme.py                # Visual styling, reusable UI helpers and Plotly theme
├── model.pkl               # Bundled pre-trained demonstration model
├── requirements.txt        # Python package dependencies
├── README.md               # Project documentation
├── .gitignore              # Files excluded from version control
│
└── .streamlit/
    └── config.toml         # Streamlit application theme/configuration
```

The `.streamlit/config.toml` file is optional for core functionality but provides the repository's Streamlit theme configuration. Keep it at the project root under the `.streamlit` directory.

---

## Technology stack

| Technology | Role |
|---|---|
| Python | Main programming language |
| Streamlit | Interactive web application and user interface |
| scikit-learn | Decision-tree models, data splitting and evaluation |
| pandas | CSV loading, tabular data preparation and analysis |
| NumPy | Numeric operations and data handling |
| Plotly | Interactive charts and visualisations |
| joblib | Serialising and loading model artefacts |

Dependencies are listed in `requirements.txt`. The repository pins **scikit-learn 1.6.1** because the bundled model was saved with that version. Model persistence across different scikit-learn versions is not guaranteed.

---

## Run the project locally

### Requirements

- Python 3.10 or 3.11 is recommended for the supplied environment.
- `pip`
- Internet access for the initial dependency installation.

### 1. Download the repository

Clone it:

```bash
git clone https://github.com/YOUR-USERNAME/modelforge-studio.git
cd modelforge-studio
```

Or use GitHub's **Code → Download ZIP** option, then extract the archive and open a terminal in the extracted project directory.

Replace `YOUR-USERNAME` with the GitHub account that owns the repository.

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the application

```bash
streamlit run app.py
```

Streamlit will print a local address, normally:

```text
http://localhost:8501
```

Open that address in your browser.

### 5. Stop the application

In the terminal where Streamlit is running, press:

```text
Ctrl + C
```

---

## Deploy on Streamlit Community Cloud

The app can be deployed from a GitHub repository using Streamlit Community Cloud.

1. Push the complete project to GitHub.
2. Make sure `app.py` and `requirements.txt` are in the repository root.
3. Make sure `model.pkl` is included if you want the bundled pre-trained model.
4. Keep `.streamlit/config.toml` in the root-level `.streamlit` directory.
5. Sign in to Streamlit Community Cloud with GitHub.
6. Choose to create a new app.
7. Select the repository, branch and `app.py` as the app's main file.
8. Deploy and wait for dependency installation and startup.
9. Open the generated app URL and test the pages.

Official deployment guide:  
https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app

If deployment fails, inspect the app logs. Dependency versions, missing files, incorrect paths and incompatible model artefacts are common causes.

---

## How the code is organised

### `app.py` — application layer

This file contains the Streamlit application.

Responsibilities include:

- Page configuration and navigation.
- Sidebar and active-model selection.
- Streamlit session-state management.
- Dataset upload controls.
- Training settings and button actions.
- Calling functions from `core.py`.
- Displaying charts, tables, metrics, forms and downloads.
- Rendering the eight application pages.

The main page functions are:

| Function | Purpose |
|---|---|
| `page_forecast()` | Weather-specific demonstration workflow |
| `page_train()` | Dataset selection, preparation controls and model training |
| `page_analytics()` | Model metrics, external evaluation and cross-validation |
| `page_tree()` | Tree diagram, feature importance and rules |
| `page_predict()` | Single-row prediction and decision path |
| `page_hub()` | Model import/export and compatibility information |
| `page_explore()` | Dataset charts and exploratory views |
| `page_report()` | Downloadable model report |

### `core.py` — machine-learning layer

This file keeps the principal data and model operations separate from the Streamlit interface.

Key functions include:

| Function | Purpose |
|---|---|
| `sample_weather_df()` | Creates the built-in synthetic sample weather dataset |
| `detect_task()` | Infers classification or regression from the target |
| `prepare_dataset()` | Selects columns, removes incomplete rows, encodes features and splits data |
| `fit_bundle()` | Fits a decision-tree classifier or regressor and builds model metadata |
| `load_model_file()` | Loads a trusted pickle/joblib decision-tree model |
| `bundle_from_model()` | Creates the app's metadata bundle around an imported model |
| `attach_evaluation()` | Attaches user-supplied labelled evaluation data |
| `evaluate()` | Calculates task-specific evaluation metrics |
| `depth_sweep()` | Trains trees at different depths for comparison |
| `recommend_depth()` | Suggests a depth from the depth-sweep results |
| `cv_report()` | Runs cross-validation when the required training data is available |
| `tree_profile()` | Calculates structural information about the decision tree |
| `leaf_info()` | Gets information about the leaf reached by an input |
| `decision_path_rows()` | Converts a prediction's tree path into readable rows |
| `tree_figure()` | Builds a visual decision-tree diagram |
| `model_bytes()` | Serialises a model for download |
| `model_card()` | Builds structured metadata for a model |
| `build_report()` | Creates a Markdown model report |
| `card_json()` | Converts a model card to formatted JSON |

### `theme.py` — presentation layer

This file contains the app's visual system and reusable presentation helpers.

It defines:

- Colour palette and Plotly colour scales.
- CSS injected into the Streamlit application.
- Responsive widget-width helpers.
- HTML escaping and rendering helpers.
- Page headers, metric strips, hero panels and advisory rows.
- Plotly chart theme configuration.

Keeping these helpers in a separate file makes the interface easier to maintain.

### `model.pkl` — bundled model artefact

This is the serialised demonstration decision tree loaded by the application. It is a binary model file, not source code. It must be kept in the expected location and loaded with a compatible scikit-learn environment.

### `requirements.txt` — dependencies

This file tells pip which Python packages and versions/ranges the application needs.

---

## Model import and export

### Export

In Model Hub, the active model can be downloaded in two forms:

**1. Pickle/joblib model (`.pkl`)**

Contains the serialised scikit-learn decision-tree estimator. It is intended for reuse in a compatible Python/scikit-learn environment.

**2. JSON model card (`.json`)**

Contains readable metadata such as:

- Model source and creation time.
- scikit-learn version.
- Task and target.
- Raw feature names.
- Class names, where applicable.
- Numeric feature ranges.
- Categorical feature values.
- Supported hyperparameter values.
- Tree depth, leaf count and training-row count.
- Selected evaluation metrics, when available.

The model card is descriptive metadata; it is not itself a model that can make predictions.

### Import

The current Model Hub accepts `.pkl` and `.joblib` files that contain a scikit-learn `DecisionTreeClassifier` or `DecisionTreeRegressor`.

The current importer does **not** provide universal support for:

- Arbitrary scikit-learn estimators.
- Complete scikit-learn preprocessing pipelines.
- TensorFlow or Keras models.
- PyTorch models.
- XGBoost, LightGBM or CatBoost models.
- ONNX models.
- Models that require custom Python classes or unavailable dependencies.

An imported estimator may not include its original training data, evaluation split, feature preprocessing metadata or original feature semantics. Supply compatible inputs and independent labelled evaluation data before relying on its predictions.

---

## Evaluation metrics

The metrics depend on the selected task.

### Regression

| Metric | Meaning |
|---|---|
| R² | Proportion of target variance explained relative to a baseline |
| RMSE | Root mean squared error; penalises larger errors more strongly |
| MAE | Mean absolute error |
| Median absolute error | Median of absolute prediction errors |

An R² value can be negative when the model performs worse than a simple mean-prediction baseline on the evaluation data. Error metrics are expressed in the target's units.

### Classification

| Metric | Meaning |
|---|---|
| Accuracy | Fraction of predictions that are correct |
| Precision (weighted) | Weighted average of class-level precision |
| Recall (weighted) | Weighted average of class-level recall |
| F1 (weighted) | Weighted average of class-level F1 scores |
| Confusion matrix | Counts of actual versus predicted classes |
| ROC AUC | Area under the ROC curve when the applicable binary-classification probabilities are available |

### Cross-validation

The analytics page can run five-fold cross-validation for a model trained in the current session, using the stored training dataset. Cross-validation is not available for an imported model when its original training data is absent.

---

## Important limitations

ModelForge Studio is a prototype and has several deliberate scope boundaries:

1. **Decision trees only:** Training currently uses scikit-learn decision-tree classifier and regressor models.
2. **Tabular CSV data:** The user-facing training workflow is designed for CSV datasets, not images, audio, video or unstructured documents.
3. **In-memory workflow:** Uploaded data and newly trained models are handled in the current Streamlit session. There is no database-backed user account or permanent model registry.
4. **Session persistence:** A model trained or imported during a session should be exported if it needs to be kept. Do not assume it will remain available in a later session.
5. **No multi-user workspace:** The app does not currently provide authentication, teams, project permissions or shared workspaces.
6. **No automatic deployment of user models:** Exporting a model does not publish it as a production prediction API.
7. **No arbitrary model support:** The importer is limited to scikit-learn decision-tree classifier/regressor objects.
8. **Preprocessing portability:** An exported estimator does not automatically package every data-cleaning and preprocessing step used before training.
9. **Weather prototype validation:** The included weather model's scientific suitability and real-world predictive performance must be independently verified before operational use.
10. **No production guarantees:** Metrics and warnings are exploratory aids, not guarantees of model quality, fairness, reliability or safety.

---

## Security and responsible use

- **Treat pickle/joblib files as executable content.** Loading an untrusted pickle or joblib file can execute arbitrary code. Only import files from sources you trust.
- Do not upload passwords, API keys, private datasets or other secrets to a public GitHub repository.
- Review uploaded data before sharing it. Remove personal, confidential or regulated information unless you have permission and a lawful basis to use it.
- Validate the origin, meaning, units and quality of all training and evaluation data.
- Keep training, validation and test data appropriately separated. Do not evaluate a model on data it has already memorised and then describe the result as independent performance.
- Do not use a prototype prediction as the sole basis for medical, financial, legal, safety-critical, agricultural or other high-impact decisions.
- Keep dependencies updated and review security advisories before deploying the app publicly.

---

## Troubleshooting

### `streamlit` is not recognised

Activate the virtual environment and install the dependencies:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

### `ModuleNotFoundError`

Make sure the virtual environment is activated and run:

```bash
pip install -r requirements.txt
```

### The pre-trained model cannot be loaded

- Confirm `model.pkl` exists in the same directory as `app.py`, or in the supported `models/` subdirectory.
- Install the pinned dependencies from `requirements.txt`.
- Check the terminal or Streamlit logs for the loading error.
- If the model file is missing or incompatible, the application may fall back to a synthetic demonstration model.

### Model version warning

The bundled model was saved with scikit-learn 1.6.1. Use the pinned dependency in `requirements.txt` when possible. scikit-learn does not guarantee that persisted models are compatible across versions.

### CSV upload or training fails

Check that:

- The file is a valid CSV with a header row.
- The selected target column exists and contains usable values.
- At least one feature is selected.
- At least 30 complete rows remain after missing rows are removed.
- Regression targets are numeric.
- A classification target has at least two classes.
- The selected feature columns contain suitable values.

### Streamlit deployment fails

Check the deployment logs and verify:

- The app entry point is `app.py`.
- `requirements.txt` is in the repository root.
- All Python modules imported by `app.py` are committed.
- `model.pkl` is present if the bundled model is required.
- `.streamlit/config.toml` is in the correct directory.
- The repository contains no accidental local paths or uncommitted files.

---

## Development and contribution

Contributions, bug reports and suggestions are welcome.

A useful contribution workflow:

1. Fork the repository.
2. Create a feature branch.
3. Make a focused change.
4. Test the application locally.
5. Check that the existing pages and bundled model still work.
6. Submit a pull request explaining the change and how it was tested.

Before contributing, consider whether a change affects:

- CSV preparation and feature encoding.
- Model bundle metadata.
- Prediction input handling.
- Model import/export compatibility.
- Streamlit session state.
- Dependency versions.
- Documentation and deployment instructions.

### Suggested future development directions

These are possible future enhancements, not features currently implemented:

- Add more supported algorithms.
- Support complete preprocessing pipelines during model export/import.
- Add persistent model and dataset storage.
- Add experiment history and model comparison.
- Add downloadable training datasets and reproducible experiment metadata.
- Add automated tests and continuous integration.
- Add authentication and multi-user project workspaces.
- Add model-serving APIs and deployment workflows.
- Add fairness, explainability and data-quality checks.

---

## License

No open-source license has been selected for this repository yet.

Until a license is added, do not assume that others have permission to redistribute, modify or commercially use the code. If you intend to publish the project as open source, choose a license and add its full text in a `LICENSE` file.

---

## Acknowledgements

ModelForge Studio is built with the Python open-source ecosystem, especially Streamlit, scikit-learn, pandas, NumPy, Plotly and joblib.

If you use or extend this project, please retain appropriate notices and follow the licenses of the dependencies and any datasets or model artefacts you include.
