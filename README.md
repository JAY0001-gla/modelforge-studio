# ModelForge Studio

### Build. Train. Test. Understand.

**ModelForge Studio** is an interactive machine-learning workbench built with Python, Streamlit and scikit-learn. It gives users one place to explore tabular data, train decision-tree models, evaluate their performance, inspect how a tree makes decisions, generate predictions and export model artefacts.

The repository includes a pre-trained **Weather Prediction** model as its first working demonstration. Users can explore that model immediately or upload their own CSV dataset and train a new decision tree.

> **Project status:** Prototype / early-stage release  
> **Current training algorithms:** Decision Tree Classifier and Decision Tree Regressor  
> **Supported training input:** Tabular CSV datasets  
> **Application framework:** Streamlit

---

## Contents

- [1. Overview](#1-overview)
- [2. First included model: Weather Prediction](#2-first-included-model-weather-prediction)
- [3. Platform features](#3-platform-features)
- [4. Getting started as a user](#4-getting-started-as-a-user)
- [5. Train a model with your own dataset](#5-train-a-model-with-your-own-dataset)
- [6. Understand the training workflow](#6-understand-the-training-workflow)
- [7. Dataset requirements and preparation](#7-dataset-requirements-and-preparation)
- [8. Model evaluation and metrics](#8-model-evaluation-and-metrics)
- [9. Model import, export and model cards](#9-model-import-export-and-model-cards)
- [10. Project structure](#10-project-structure)
- [11. Codebase guide](#11-codebase-guide)
- [12. Technology stack](#12-technology-stack)
- [13. Installation and local setup](#13-installation-and-local-setup)
- [14. Deploy on Streamlit Community Cloud](#14-deploy-on-streamlit-community-cloud)
- [15. Configuration](#15-configuration)
- [16. Limitations and current scope](#16-limitations-and-current-scope)
- [17. Security and responsible use](#17-security-and-responsible-use)
- [18. Troubleshooting](#18-troubleshooting)
- [19. Development and contribution](#19-development-and-contribution)
- [20. License](#20-license)

---

## 1. Overview

Machine-learning experimentation often involves switching between notebooks, scripts, charts and model files. ModelForge Studio brings several common steps into a single interactive application.

With the current version, a user can:

1. Open the included Weather Prediction model and try it without first training a model.
2. Upload a CSV dataset.
3. Inspect the dataset and apply basic cleaning operations.
4. Select the target column and input features.
5. choose a classification or regression task.
6. Configure decision-tree hyperparameters.
7. Train a model and compare training and held-out test performance.
8. Explore feature importance, tree structure and decision paths.
9. Enter new feature values and generate predictions.
10. Evaluate a model on a separate labelled CSV.
11. Export a trained model, its JSON model card and a Markdown report.

ModelForge Studio is intended as a practical learning and experimentation environment. It is not currently a universal AutoML service or a production model-hosting platform.

---

## 2. First included model: Weather Prediction

### Model 001 — Weather Prediction

The repository includes `model.pkl`, a pre-trained scikit-learn decision-tree model. It is loaded automatically when the application starts, provided the file is present and compatible with the installed dependencies.

| Item | Description |
|---|---|
| Demonstration model | Weather Prediction |
| Model family | Decision Tree |
| Task expected by the Weather Forecast page | Regression |
| Prediction shown | Temperature estimate |
| Display unit | °C |
| Model file | `model.pkl` |
| Framework | scikit-learn |
| Purpose | Demonstrate loading, prediction, tree exploration and model analytics |

The Weather Forecast page provides six atmospheric input controls, preset conditions and a prediction workflow. After a forecast is run, users can inspect the decision path and explore a what-if chart by varying one numeric input while keeping the other inputs fixed.

**Model interpretation note:** The application assigns the bundled model the target label `Temperature` and unit `°C`. The original data provenance, feature definitions, measurement units and independent real-world validation should be checked before interpreting the output as an operational weather forecast. A prediction from this prototype should be treated as an educational demonstration, not an authoritative meteorological forecast.

If the bundled model is missing or cannot be loaded, the app falls back to a demonstration model generated from synthetic weather data. Synthetic data is for demonstrating the workflow; it is not real-world evidence.

The bundled model is described in the project as a deep/unpruned tree. Users can train a new tree with a depth limit and other regularisation settings in Train Studio.

---

## 3. Platform features

The application has eight pages. The active model can be switched in the sidebar between **Pre-trained model** and **My trained model**.

### 3.1 Weather Forecast

A focused demonstration of the included weather regression model.

- Accepts six atmospheric readings through input controls.
- Provides Cool day, Typical day and Warm day presets.
- Runs the active pre-trained model to estimate temperature.
- Displays a temperature band and illustrative guidance.
- Shows the decision path followed through the tree.
- Includes a what-if chart for changing one numeric input.
- Allows the user to adjust the temperature-band thresholds used for the guidance display.

The page expects the default model to be a regression tree. If the loaded model is a classifier, the app explains that the user should use Prediction Center for that model instead.

### 3.2 Train Studio

The main workspace for creating a model from a user's own data.

- Starts with the built-in synthetic sample weather dataset or accepts a CSV upload.
- Displays a data preview, row count, column count, missing-value count and duplicate-row count.
- Offers basic cleaning actions:
  - Drop rows with missing values.
  - Fill numeric missing values with their column median.
  - Remove duplicate rows.
- Lets the user choose a target column and input features.
- Offers Auto-detect, Regression and Classification task choices.
- Provides tree settings including maximum depth, minimum samples per leaf, minimum samples to split, split criterion and pruning strength (`ccp_alpha`).
- Can compare different tree depths and suggest a depth using the held-out test score.
- Trains and activates the new model for the current session.
- Shows training and test scores, tree size and a basic overfitting warning.

### 3.3 Model Analytics

A workspace for reviewing the active model and testing it against labelled data.

Depending on task and available data, the page can show:

- Regression metrics and actual-versus-predicted / residual visualisations.
- Classification metrics and a confusion matrix.
- Evaluation using a user-uploaded CSV containing the required features and true target.
- Five-fold cross-validation for models trained in the current session, when the stored training data is available.
- A learning-curve view to help compare training and validation behaviour.

An imported model does not automatically contain its original training dataset. Cross-validation and learning-curve analysis that require original training data may therefore be unavailable for imported models. A separate labelled evaluation CSV can still be supplied when its columns match the model's expected inputs and target.

### 3.4 Tree Explorer

A visual explanation area for the active decision tree.

- Shows tree depth, leaf count and median leaf size.
- Ranks the model's feature importances.
- Displays a decision-tree diagram with a selectable display depth.
- Visualises how training records are distributed across leaves.
- Displays readable text rules from the tree.
- Provides a structural overfitting-risk indicator.

Feature importance is based on the fitted tree's impurity reduction. It describes how the model used features; it does not establish causation or prove that a feature is important outside this particular fitted model.

### 3.5 Prediction Center

A general single-record prediction interface for the active supported decision-tree model.

- Builds input controls from the model's feature metadata.
- Accepts values for one prediction.
- Displays a predicted class or numeric value.
- Displays class probabilities when available for classification.
- Shows the decision path through the tree.
- Warns when numeric inputs fall outside ranges inferred from the model's split thresholds.

An out-of-range warning is a diagnostic hint, not a guarantee that a prediction is invalid.

### 3.6 Model Hub

The place to import and export supported models.

- Downloads the active estimator as a `.pkl` file.
- Downloads a JSON model card containing model metadata.
- Accepts `.pkl` and `.joblib` uploads.
- Checks that an imported estimator is a supported scikit-learn decision-tree classifier or regressor.
- Shows the scikit-learn version used by the running application.

Only import model files from trusted sources. Pickle/joblib loading can execute code.

### 3.7 Data Explorer

Exploratory visualisations for the current working dataset.

- Numeric correlation heatmap.
- Histograms and box plots.
- Two-dimensional scatter plots.
- Optional colour grouping.
- Optional linear trend line for suitable numeric relationships.
- Three-dimensional scatter plot when at least three numeric columns are available.
- Missing-value overview.

The working dataset is the sample dataset or the CSV currently selected in Train Studio. Data Explorer is descriptive; it does not automatically establish statistical significance or causal relationships.

### 3.8 Report

Creates a downloadable Markdown report for the active model.

The report can include:

- Model source, task and target.
- Input feature names and training-row count.
- Tree depth, number of leaves and median leaf size.
- Structural overfitting-risk estimate.
- Evaluation metrics, when evaluation data is available.
- Feature-importance values.
- Model hyperparameters.

---

## 4. Getting started as a user

You do not need to train a model before trying the application.

### Explore the included model

1. Start ModelForge Studio.
2. In the sidebar, keep **Pre-trained model** selected.
3. Open **Weather Forecast**.
4. Enter the atmospheric readings or choose one of the presets.
5. Click **Run forecast**.
6. Review the estimated temperature and guidance.
7. Open the **Decision path** tab to see the tree decisions.
8. Open the **What-if** tab to see how changing one input affects the tree's output.
9. Visit **Tree Explorer**, **Model Analytics**, **Prediction Center** and **Model Hub** to explore other model tools.

### Switch between models

The sidebar's **Active model** selector switches between:

- **Pre-trained model** — the bundled demonstration model.
- **My trained model** — a model trained or imported during the current session.

Train a model in Train Studio or import one in Model Hub before selecting My trained model.

---

## 5. Train a model with your own dataset

This is the main workflow for users who want to experiment with their own data.

### Step 1 — Prepare a CSV

Prepare a tabular CSV file with column headers. Include the value you want to predict and the input columns that may help predict it.

Example structure:

| Hours_Studied | Attendance | Practice_Tests | Final_Score |
|---:|---:|---:|---:|
| 2 | 70 | 1 | 48 |
| 4 | 82 | 3 | 66 |
| 6 | 90 | 5 | 84 |
| 3 | 75 | 2 | 55 |

In this example, `Final_Score` could be the target and the other columns could be the features. This is only an illustrative example, not a dataset shipped with the project.

### Step 2 — Upload the CSV

1. Open **Train Studio**.
2. Under Dataset, select **Upload CSV**.
3. Choose your `.csv` file.
4. Review the preview and dataset statistics.

### Step 3 — Clean the data

Use the available cleaning controls if appropriate:

- **Drop rows with missing values** removes rows containing missing cells.
- **Fill numeric gaps with the median** fills missing numeric values using each numeric column's median.
- **Remove duplicate rows** removes duplicate records.

These actions update the working dataset in the current session. Review the result before training. The training preparation function also drops rows that have missing values in the selected target or selected feature columns.

### Step 4 — Select the target and features

- **Target to predict:** the column the model should learn to predict.
- **Input features:** the columns the model can use to make its prediction.

Do not include information that would only be known after the target outcome occurs. Avoid leakage columns such as a final result copied into an input field.

### Step 5 — Select the task

Choose one of:

- **Auto-detect**
- **Regression**
- **Classification**

Auto-detection is a heuristic based on the target column's data type and number of distinct values. Review the detected task and override it if necessary.

### Step 6 — Configure the decision tree

The available settings include:

| Setting | What it controls |
|---|---|
| Unlimited depth | Allows the tree to grow without a configured maximum depth |
| Max depth | Maximum number of split levels |
| Held out for testing (%) | Portion of data reserved for a test split |
| Min samples per leaf | Minimum training records required in a leaf |
| Min samples to split | Minimum records required before a node can split |
| Split criterion | Measure used to choose a split |
| Pruning strength (`ccp_alpha`) | Cost-complexity pruning parameter; larger values apply stronger pruning |

Classification criteria include Gini, entropy and log loss. Regression criteria include squared error and Friedman MSE.

### Step 7 — Suggest a depth (optional)

Click **Suggest a depth** to train and compare trees at different depths using the current prepared dataset and test split. Review the training and test score chart and use the suggested depth if it is suitable.

A suggested depth is a convenience, not a guarantee of optimal generalisation. It should be validated on data appropriate to the real task.

### Step 8 — Train

Click **Train model**. The app prepares the selected data, fits the decision tree and activates the result as **My trained model** for the current session.

### Step 9 — Explore the result

- **Model Analytics:** review test performance and overfitting indicators.
- **Tree Explorer:** inspect splits, rules and feature importance.
- **Prediction Center:** enter a new record and request a prediction.
- **Model Hub:** download the estimator and model card.
- **Report:** download a Markdown report.

---

## 6. Understand the training workflow

The primary data preparation and model operations are implemented in `core.py`.

### 6.1 Column selection

The user chooses one target column and one or more feature columns. The target is not used as an input feature.

### 6.2 Missing values and data types

The preparation workflow:

- Selects the target and chosen feature columns.
- Removes rows with missing values in those selected columns.
- Converts boolean features to integer values.
- Treats non-numeric feature columns as categorical.
- Converts categorical values to strings.
- One-hot encodes categorical features using pandas.
- Stores feature metadata used by the app to construct prediction inputs.

The median-fill control is a separate user-triggered dataset-cleaning action. It is not a fitted imputation pipeline automatically packaged with the exported estimator.

### 6.3 Task detection

The automatic task detector uses a simple heuristic:

- Non-numeric or boolean targets are treated as classification.
- Numeric targets with integer-like values and no more than 10 unique values are treated as classification.
- Other numeric targets are treated as regression.

This heuristic may not match the meaning of every dataset. For example, category IDs stored as numbers may need classification, while a numeric quantity with few observed values may still be a regression problem. Select the task deliberately when needed.

### 6.4 Encoding

Categorical input features are one-hot encoded. Classification target labels are encoded for fitting and translated back for presentation where the app has the relevant class metadata.

### 6.5 Train/test split

The application uses scikit-learn's `train_test_split` with a fixed random seed of `42` and the test percentage selected by the user. For classification, it attempts to stratify the split when the class counts allow it.

The current preparation function requires at least 30 usable rows after selected-column missing rows are removed.

### 6.6 Model fitting

The application fits one of the following:

- `DecisionTreeClassifier` for classification.
- `DecisionTreeRegressor` for regression.

The model is fitted using the configured tree hyperparameters.

### 6.7 Performance review

The Train Studio page shows training and held-out test scores, tree structure information and a basic train/test gap warning. A large gap can indicate overfitting.

A single random split is only one estimate of performance. It does not guarantee that the model will perform similarly on future, unseen or differently distributed data.

---

## 7. Dataset requirements and preparation

### Accepted file type

- CSV (`.csv`) for the training and evaluation upload workflows.

### Recommended dataset format

- Include a header row with unique, meaningful column names.
- Include one target column.
- Include at least one input feature.
- Provide at least 30 usable rows after selected-column missing rows are removed.
- Use a numeric target for regression.
- For classification, provide at least two target classes with enough examples for a train/test split.
- Keep feature names, units, meanings and data types consistent between training and later evaluation or prediction.

### Categorical values

Non-numeric feature columns are treated as categorical and one-hot encoded during training. The app records category metadata for its own prediction workflow.

For imported external estimators, the app does not reconstruct an arbitrary external preprocessing pipeline. An estimator trained on one-hot encoded or otherwise transformed data may require exactly compatible input preparation. A plain decision-tree estimator alone does not necessarily contain all preprocessing steps used before it was fitted.

### Evaluation CSV

For evaluation, upload a labelled CSV containing:

- Every feature column required by the active model.
- The true target column to compare with predictions.

The evaluation columns must match the model's expected feature representation and semantics. Evaluation data should be separate from the data used to fit the model whenever possible.

### Dataset size

Uploaded CSV files are loaded into memory using pandas. Very large files may take longer to load and train and may require substantial memory. The current application does not provide distributed or streaming training.

---

## 8. Model evaluation and metrics

The available metrics depend on the model task.

### Regression metrics

| Metric | Interpretation |
|---|---|
| R² | Compares the model with a baseline that predicts the mean target value. Can be negative if the model performs worse than that baseline. |
| RMSE | Root mean squared error; larger errors have greater influence. |
| MAE | Mean absolute error in the target's units. |
| Median absolute error | Median absolute difference between predictions and true values. |

### Classification metrics

| Metric | Interpretation |
|---|---|
| Accuracy | Fraction of evaluated predictions that are correct. |
| Weighted precision | Class precision averaged with class support as weights. |
| Weighted recall | Class recall averaged with class support as weights. |
| Weighted F1 | Weighted average of class-level F1 scores. |
| Confusion matrix | Counts correct and incorrect predictions by actual and predicted class. |
| ROC AUC | Shown when the applicable probability outputs and classification setup support its calculation. |

### Cross-validation and learning curves

The app can perform five-fold cross-validation for a model trained in the current session when its training data is available. Imported models generally do not include the original training data in this application's model bundle, so those analyses may not be available.

Do not use the same data for training and final evaluation and then describe the result as independent test performance.

---

## 9. Model import, export and model cards

### Export a model

1. Select the model you want in the sidebar.
2. Open **Model Hub**.
3. Under Export, choose **Download model (.pkl)**.
4. Optionally choose **Download model card (.json)**.

The `.pkl` file stores the estimator. The JSON model card is a readable description of the model and is not itself executable or a replacement for the estimator.

### Model card contents

The model card can record:

- Model source and creation timestamp.
- scikit-learn version.
- Task and target.
- Raw feature names and class names, where applicable.
- Numeric feature ranges and categorical values.
- Model hyperparameters.
- Tree depth, leaf count and training-row count.
- Selected evaluation metrics, when available.

### Import a model

1. Open **Model Hub**.
2. Under Import, select a `.pkl` or `.joblib` file.
3. Click **Activate imported model**.
4. If loading succeeds, the model becomes available as **My trained model**.
5. Supply labelled evaluation data in Model Analytics if you want to score it on a separate dataset.

### Supported imported estimators

The importer accepts scikit-learn:

- `DecisionTreeClassifier`
- `DecisionTreeRegressor`

It does not currently provide universal import support for arbitrary scikit-learn estimators, full preprocessing pipelines, TensorFlow/Keras, PyTorch, XGBoost, LightGBM, CatBoost, ONNX or custom model formats.

### Version compatibility

The bundled model was saved with **scikit-learn 1.6.1**. The repository pins that version in `requirements.txt`. Persisted scikit-learn models are not guaranteed to be compatible across library versions; use the pinned environment when possible.

---

## 10. Project structure

```text
modelforge-studio/
│
├── .streamlit/
│   └── config.toml       # Streamlit theme, toolbar, upload and browser settings
│
├── app.py                # Streamlit application entry point and page UI
├── core.py               # Data preparation, model training and evaluation logic
├── theme.py              # CSS, reusable UI components and Plotly theme
├── model.pkl             # Pre-trained Weather Prediction model
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
├── LICENSE               # MIT license
└── .gitignore             # Files excluded from Git
```

Keep `model.pkl` in the same directory as `app.py`, or in the supported `models/model.pkl` location. Keep the `.streamlit` directory at the repository root.

---

## 11. Codebase guide

### `app.py` — application and interface

This is the Streamlit entry point. It is responsible for:

- Page configuration and navigation.
- Sidebar branding and active-model selection.
- Streamlit session state.
- Dataset upload and cleaning controls.
- Training configuration and user actions.
- Calling machine-learning functions from `core.py`.
- Rendering metrics, charts, tables, forms and downloads.
- Connecting the eight application pages.

Main page functions:

| Function | Responsibility |
|---|---|
| `page_forecast()` | Weather model forecast, guidance, decision path and what-if view |
| `page_train()` | Dataset upload, cleaning, feature selection, configuration and training |
| `page_analytics()` | Model metrics, external evaluation and generalisation analysis |
| `page_tree()` | Tree profile, feature importance, diagram and text rules |
| `page_predict()` | Single-record prediction and decision path |
| `page_hub()` | Model import, export and compatibility information |
| `page_explore()` | Dataset exploration charts |
| `page_report()` | Download and preview the model report |

Other helpers manage chart rendering, tables, page navigation, notifications, model selection and prediction input controls.

### `core.py` — machine-learning and data layer

This module contains the core data and model operations, separate from the Streamlit page layout.

Important functions include:

| Function | Responsibility |
|---|---|
| `sample_weather_df()` | Creates the built-in synthetic weather dataset |
| `detect_task()` | Suggests classification or regression from the target column |
| `prepare_dataset()` | Selects columns, handles selected data types, encodes features and splits data |
| `fit_bundle()` | Fits a decision-tree model and packages model/session metadata |
| `demo_bundle()` | Builds the fallback demonstration model when the bundled model cannot be loaded |
| `bundle_from_model()` | Creates the application's metadata bundle around a supported imported tree |
| `load_model_file()` | Loads a trusted pickle/joblib decision-tree estimator |
| `evaluate()` | Calculates task-specific evaluation metrics |
| `attach_evaluation()` | Prepares user-supplied labelled evaluation data |
| `depth_sweep()` | Compares models trained with different maximum depths |
| `recommend_depth()` | Suggests a depth from the sweep results |
| `cv_report()` | Runs cross-validation when training data is available |
| `tree_profile()` | Calculates tree depth, leaves and structural information |
| `leaf_info()` | Inspects the leaf reached by an input |
| `decision_path_rows()` | Converts a tree decision path into readable rows |
| `tree_figure()` | Builds a visual decision-tree diagram |
| `encode_frame()` | Converts raw prediction inputs into the model's expected feature representation |
| `out_of_range()` | Checks numeric inputs against inferred feature ranges |
| `what_if_curve()` | Calculates predictions while varying one input |
| `model_bytes()` | Serialises a model for download |
| `model_card()` | Builds model metadata for JSON export |
| `build_report()` | Creates a Markdown model report |
| `card_json()` | Serialises a model card as formatted JSON |

### `theme.py` — design system and reusable components

This module provides the visual presentation layer:

- Colour palette and typography settings.
- CSS for the page canvas, sidebar, buttons, tabs, metrics and responsive layout.
- Animated weather/forecast hero component.
- Reusable page headers, metric strips, guidance rows and callouts.
- HTML escaping and rendering helpers.
- Streamlit widget width compatibility helpers.
- Plotly chart template registration.

The design uses a light interface with blue/cobalt and teal accents, subtle borders and responsive layouts. Motion is reduced when the user's operating-system preference requests reduced motion.

### `model.pkl` — pre-trained model artefact

A binary serialised scikit-learn decision-tree model used by the first demonstration workflow. It is not Python source code. Keep it in the expected path and use the pinned dependency version.

### `requirements.txt` — dependencies

Lists the Python packages needed to run the application. The scikit-learn version is pinned to match the bundled model.

### `.streamlit/config.toml` — Streamlit configuration

The repository configuration sets the light theme, primary and background colours, minimal toolbar mode, maximum upload size and disables browser usage-stat collection.

---

## 12. Technology stack

| Technology | Use in this project |
|---|---|
| Python | Application and model logic |
| Streamlit | Interactive browser-based interface |
| scikit-learn | Decision-tree models, data splitting, preprocessing utilities and metrics |
| pandas | CSV loading, data cleaning and tabular operations |
| NumPy | Numerical operations |
| Plotly | Interactive charts and visualisations |
| joblib | Serialising and loading model artefacts |

The dependency versions/ranges are maintained in `requirements.txt`.

---

## 13. Installation and local setup

### Prerequisites

- Python **3.10–3.13** is the documented supported range for this project.
- `pip`
- Git (optional, if cloning the repository).
- Internet access for installing packages.

### Option A — Clone with Git

```bash
git clone https://github.com/JAY0001-gla/modelforge-studio.git
cd modelforge-studio
```

### Option B — Download ZIP

1. Open the GitHub repository.
2. Click **Code**.
3. Select **Download ZIP**.
4. Extract the downloaded ZIP.
5. Open a terminal in the extracted `modelforge-studio` folder.

### Create a virtual environment

A virtual environment keeps this project's dependencies separate from other Python projects.

**Windows PowerShell:**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell does not allow activation, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install the dependencies

With the virtual environment activated, run:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Run ModelForge Studio

```bash
python -m streamlit run app.py
```

The terminal will display a local address. Open:

```text
http://localhost:8501
```

The application runs in your browser while the Streamlit process remains active in the terminal.

### Stop the application

Press `Ctrl + C` in the terminal where Streamlit is running.

### Run without Git

If you downloaded the ZIP, you do not need Git to run the app. Extract it, open a terminal in the project folder, create/activate the virtual environment, install requirements and run Streamlit as described above.

---

## 14. Deploy on Streamlit Community Cloud

ModelForge Studio can be deployed from its GitHub repository using Streamlit Community Cloud.

1. Push the complete project to GitHub.
2. Confirm that `app.py` and `requirements.txt` are in the repository root.
3. Confirm that `core.py`, `theme.py` and `model.pkl` are committed.
4. Confirm that `.streamlit/config.toml` is in the root-level `.streamlit` folder.
5. Sign in to Streamlit Community Cloud using GitHub.
6. Choose **Create app**.
7. Select this repository, the `main` branch and `app.py` as the main file.
8. Deploy the application.
9. Wait for package installation and app startup.
10. Open the generated Streamlit app URL and test the pages.

Official guide:  
https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app

If deployment fails, check the deployment logs for missing modules, incorrect file paths, dependency conflicts or model-loading errors.

---

## 15. Configuration

The project uses `.streamlit/config.toml` for Streamlit configuration.

Current settings include:

- Light theme.
- Primary colour: `#2A5BD7`.
- Page background: `#F4F7FB`.
- Secondary background: `#FFFFFF`.
- Text colour: `#1F2A3C`.
- Minimal toolbar mode.
- Maximum upload size: 200 MB.
- Browser usage statistics collection disabled.

If you change the theme or upload limit, edit `.streamlit/config.toml` and restart the application.

The upload limit is a maximum accepted file size, not a promise that very large CSV files will fit in the available memory or train quickly.

---

## 16. Limitations and current scope

ModelForge Studio is an early-stage project. The following boundaries describe the current implementation:

1. **Decision-tree training:** Train Studio currently trains scikit-learn decision-tree classifiers and regressors.
2. **Tabular datasets:** The training workflow is designed for CSV data, not images, audio, video or document datasets.
3. **Supported model imports:** Model Hub accepts supported scikit-learn decision-tree classifier and regressor objects, not arbitrary ML models.
4. **Preprocessing portability:** Exporting the estimator does not automatically package every cleaning and encoding operation as a complete reusable preprocessing pipeline.
5. **Session-based work:** Newly trained or imported models and working datasets are managed in the Streamlit session. Export artefacts you want to keep.
6. **No account system:** There are no user accounts, teams, access controls or shared workspaces.
7. **No persistent model registry:** The application does not provide a database-backed library of every model a user has trained.
8. **No hosted inference API:** Exporting a model does not automatically deploy it as a production API.
9. **In-memory CSV processing:** Large datasets can consume significant memory.
10. **Weather model validation:** The bundled model is a demonstration artefact. Its data provenance and real-world performance must be independently verified before operational use.
11. **Metrics are not guarantees:** A high score on one split does not prove that a model is reliable on future data.

The project can be extended with more algorithms, complete preprocessing pipelines, experiment tracking, persistent storage, user workspaces and model-serving features in future versions. Those are future directions, not claims about the current release.

---

## 17. Security and responsible use

- **Only load trusted `.pkl` and `.joblib` files.** These formats can execute arbitrary code during deserialisation. Never import a model file from an unknown or untrusted source.
- Do not commit API keys, passwords, tokens, private certificates or other secrets to a public repository.
- Do not upload personal, confidential or regulated data unless you have permission and an appropriate lawful basis.
- Review a dataset before uploading it to a hosted instance. The app processes uploaded data in memory.
- Check dataset provenance, feature definitions, units, quality and potential bias.
- Keep training and final evaluation data separate.
- Avoid target leakage: do not use inputs that reveal the answer or would not be available at prediction time.
- Do not use this prototype as the sole basis for medical, financial, legal, employment, safety-critical or other high-impact decisions.
- Review dependency updates and security advisories before public deployment.

---

## 18. Troubleshooting

### `streamlit` is not recognised

Activate the virtual environment and install the dependencies:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

### `ModuleNotFoundError`

Make sure the correct virtual environment is active, then run:

```bash
pip install -r requirements.txt
```

### The pre-trained model cannot be loaded

- Confirm `model.pkl` is next to `app.py`, or in `models/model.pkl`.
- Install the pinned dependencies from `requirements.txt`.
- Check the terminal or Streamlit logs for the underlying error.
- If the model is missing or incompatible, the app may use its synthetic fallback demonstration model.

### scikit-learn version warning

The bundled model was saved with scikit-learn 1.6.1. Keep `scikit-learn==1.6.1` from `requirements.txt` when possible. Scikit-learn does not guarantee persisted-model compatibility across versions.

### CSV upload fails

- Confirm the file is a valid `.csv`.
- Check that the file has a header row.
- Check for malformed rows, unusual encodings or delimiter issues.
- Try saving the dataset as UTF-8 CSV from your spreadsheet application.
- Confirm the file size is within the configured upload limit.

### Training fails

Check that:

- A target column is selected.
- At least one input feature is selected.
- At least 30 usable rows remain after required rows are removed.
- The target values are suitable for the selected task.
- Regression targets are numeric.
- Classification has enough examples in its classes for the selected split.
- The dataset does not contain unexpected or unsupported values.

### Imported model fails

- Confirm that the file is a trusted `.pkl` or `.joblib`.
- Confirm that it contains a scikit-learn `DecisionTreeClassifier` or `DecisionTreeRegressor`.
- Check the scikit-learn version used to save it.
- Remember that a raw estimator may not include its original preprocessing pipeline.

### Streamlit Community Cloud deployment fails

Check the deployment logs and verify:

- The entry point is `app.py`.
- `requirements.txt` is at the repository root.
- `core.py` and `theme.py` are committed.
- `model.pkl` is committed if you want the bundled model.
- `.streamlit/config.toml` is in the correct root-level folder.
- No local-only file paths or uncommitted dependencies are required.

---

## 19. Development and contribution

Suggestions, bug reports and contributions are welcome.

A typical contribution workflow:

1. Fork the repository.
2. Create a feature branch.
3. Make a focused change.
4. Install dependencies in a virtual environment.
5. Run the app locally.
6. Test the affected workflow and check that existing pages still work.
7. Submit a pull request describing the change and how it was tested.

When modifying the code, consider the interaction between:

- Dataset preparation and feature encoding.
- Model bundle metadata.
- Prediction input handling.
- Model import/export compatibility.
- Streamlit session state.
- Dependency versions.
- Documentation and deployment configuration.

### Possible future improvements

These are ideas for future development and are not part of the current implementation:

- Additional algorithms such as random forests, gradient boosting and other supervised models.
- Reusable preprocessing pipelines for import and export.
- Experiment history and model comparison.
- Persistent datasets and model storage.
- User accounts and collaborative workspaces.
- Automated tests and continuous integration.
- Model-serving APIs and deployment workflows.
- Additional explainability, data-quality and fairness tools.

---

## 20. License

This project is distributed under the **MIT License**. See the [`LICENSE`](LICENSE) file for the full license text.

---

## Acknowledgements

Built with the Python open-source ecosystem, including Streamlit, scikit-learn, pandas, NumPy, Plotly and joblib.

If you use, modify or extend this project, follow the MIT License and the licences and notices of the third-party libraries and datasets you use.
