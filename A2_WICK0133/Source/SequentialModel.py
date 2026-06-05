# Student Name: Zechariah Wicks
# Student FAN: wick0133
# File: SequentialModel.py
# Date: 04-06-2026
# Description: Sequential HMM pipeline modelling latent volatility regimes in XEJ using cloud cover and stock data.
# Usage: (ensure the virtual environment is active, and libraries installed via pip) python3 SequentialModel.py
# Licence: GNU General Public License v3.0

import os
import warnings
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm as spnorm

warnings.filterwarnings('ignore')

# Global Configuration 
OUTPUT_DIR = "./Misc/Outputs/"
CACHE_FILE = f"{OUTPUT_DIR}cachedSequentialData.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

START_DATE = "2018-01-01"
END_DATE = "2025-01-01"
MAX_STATES_SEARCH = 6
FINAL_N_STATES = 3
RANDOM_SEED = 42



# DATA INGESTION
def fetchFinancialData(ticker="^AXEJ", startDate=START_DATE, endDate=END_DATE):
    print(f"Fetching financial data for {ticker} ({startDate} to {endDate})...")
    stockData = yf.download(ticker, start=startDate, end=endDate, progress=False)

    if isinstance(stockData.columns, pd.MultiIndex):
        stockData.columns = stockData.columns.get_level_values(0)

    stockData = stockData.reset_index()
    stockData['date'] = pd.to_datetime(stockData['Date']).dt.date
    stockData['xejVolatility'] = (stockData['High'] - stockData['Low']).squeeze()

    return stockData[['date', 'xejVolatility']]


def fetchWeatherData(latitude=-34.9285, longitude=138.6007, startDate=START_DATE, endDate=END_DATE):
    print("Fetching weather data from Open-Meteo API...")
    apiUrl = "https://archive-api.open-meteo.com/v1/archive"
    apiParams = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": startDate,
        "end_date": endDate,
        "daily": "cloudcover_mean",
        "timezone": "Australia/Adelaide"
    }

    apiResponse = requests.get(apiUrl, params=apiParams, timeout=60)
    apiResponse.raise_for_status()
    weatherJson = apiResponse.json()

    weatherData = pd.DataFrame({
        'date': pd.to_datetime(weatherJson['daily']['time']).date,
        'cloudCover': weatherJson['daily']['cloudcover_mean']
    })

    return weatherData


# # Fake Data Function: uncomment to bypass the Open-Meteo API when offline.
# def fetchWeatherData(latitude=-34.9285, longitude=138.6007,
#                      startDate=START_DATE, endDate=END_DATE):
#     print("Open-Meteo API offline — generating synthetic mock weather data...")
#     dateRange = pd.date_range(start=startDate, end=endDate)
#     syntheticCloudCover = np.random.normal(loc=46.85, scale=27.82, size=len(dateRange))
#     syntheticCloudCover = np.clip(syntheticCloudCover, 0, 100)
#     return pd.DataFrame({'date': dateRange.date, 'cloudCover': syntheticCloudCover})


def preprocessData():
    if os.path.exists(CACHE_FILE):
        print("Loading sequential dataset from local cache...")
        cleanData = pd.read_csv(CACHE_FILE, parse_dates=['date'])
        cleanData['date'] = cleanData['date'].dt.date
        return cleanData

    stockData = fetchFinancialData()
    weatherData = fetchWeatherData()

    mergedData = pd.merge(stockData, weatherData, on='date')
    cleanData = mergedData.dropna()

    cleanData.to_csv(CACHE_FILE, index=False)
    print(f"Data cached to {CACHE_FILE}. Total records: {len(cleanData)}")

    return cleanData



# FEATURE PREPARATION
def prepareFeatures(cleanData):
    featureMatrix = cleanData[['xejVolatility', 'cloudCover']].values
    scaler = StandardScaler()
    scaledFeatures = scaler.fit_transform(featureMatrix)
    return scaledFeatures, scaler



# MODEL ORDER SELECTION
def selectModelOrder(scaledFeatures, maxStates=MAX_STATES_SEARCH):
    print(f"\n[Model Selection] Evaluating 2 to {maxStates} hidden states")
    results = []
    T, D = scaledFeatures.shape

    for n in range(2, maxStates + 1):
        candidateModel = hmm.GaussianHMM(
            n_components=n, covariance_type='full',
            n_iter=300, random_state=RANDOM_SEED
        )
        candidateModel.fit(scaledFeatures)
        logLikelihood = candidateModel.score(scaledFeatures)

        nParams = n * (n - 1) + (n - 1) + n * D + n * D * (D + 1) // 2
        bic = -2 * logLikelihood + nParams * np.log(T)

        results.append({
            'nStates': n, 'logLikelihood': logLikelihood,
            'bic': bic, 'nParams': nParams
        })
        print(f"  n={n}: LogL={logLikelihood:.2f}, BIC={bic:.2f}, params={nParams}")

    resultsDF = pd.DataFrame(results)
    bestN = int(resultsDF.loc[resultsDF['bic'].idxmin(), 'nStates'])
    print(f"  --> BIC-optimal states: {bestN}")

    return resultsDF, bestN



# HMM TRAINING AND INFERENCE
def trainHMM(scaledFeatures, nStates=FINAL_N_STATES):
    print(f"\n[Training] Fitting {nStates}-state Gaussian HMM")
    model = hmm.GaussianHMM(
        n_components=nStates, covariance_type='full',
        n_iter=1000, random_state=RANDOM_SEED, tol=1e-6
    )
    model.fit(scaledFeatures)
    finalLogL = model.score(scaledFeatures)
    print(f"  Training complete. Final log-likelihood: {finalLogL:.4f}")
    return model


def labelStates(model, scaler):
    originalMeans = scaler.inverse_transform(model.means_)
    volatilityMeans = originalMeans[:, 0]
    rankOrder = np.argsort(volatilityMeans)   # ascending: lowest vol first

    labels = ['Low Volatility', 'Normal Volatility', 'High Volatility']
    colours = ['#2196F3', '#FFC107', '#F44336']   # Blue, Amber, Red

    stateMap = {int(rankOrder[i]): labels[i] for i in range(model.n_components)}
    stateColourMap = {int(rankOrder[i]): colours[i] for i in range(model.n_components)}

    return stateMap, stateColourMap, originalMeans


def decodeWithViterbi(model, scaledFeatures):
    print("\n[Viterbi] Decoding optimal state sequence...")
    logProb, stateSequence = model.decode(scaledFeatures, algorithm='viterbi')
    print(f"  Viterbi total log-probability: {logProb:.4f}")
    return logProb, stateSequence


def computeForwardBackward(model, scaledFeatures):
    print("[Forward-Backward] Computing posterior state probabilities...")
    _, posteriors = model.score_samples(scaledFeatures)
    return posteriors



# EVALUATION
def evaluateModel(model, scaledFeatures, stateSequence, stateMap, resultsDF):
    print("\n" + "=" * 60)
    print("  MODEL EVALUATION SUMMARY")
    print("=" * 60)

    logLikelihood = model.score(scaledFeatures)
    T = len(stateSequence)

    print(f"  Log-Likelihood (total)     : {logLikelihood:.4f}")
    print(f"  Log-Likelihood (per sample): {logLikelihood / T:.6f}")
    print(f"  Observation Count          : {T}")
    print(f"  BIC-optimal state count    : {int(resultsDF.loc[resultsDF['bic'].idxmin(), 'nStates'])}")
    print(f"  Final model state count    : {model.n_components} (per design spec)")

    print("\n  State Occupancy:")
    for state in sorted(stateMap.keys()):
        count = int(np.sum(stateSequence == state))
        print(f"    [{stateMap[state]:20s}] {count:4d} days  ({100 * count / T:.1f}%)")

    print("\n  Initial State Distribution (π):")
    for i in range(model.n_components):
        print(f"    {stateMap[i]:20s}: {model.startprob_[i]:.4f}")

    print("\n  Transition Probability Matrix (A):")
    transDF = pd.DataFrame(
        np.round(model.transmat_, 4),
        index=[stateMap[i] for i in range(model.n_components)],
        columns=[stateMap[i] for i in range(model.n_components)]
    )
    print(transDF.to_string())

    print("\n  Emission Parameters (original scale):")
    originalMeans = model.means_ 
    for i in range(model.n_components):
        print(f"    {stateMap[i]:20s}: mean_volatility={originalMeans[i, 0]:.4f} "
              f"(scaled), mean_cloud={originalMeans[i, 1]:.4f} (scaled)")

    return logLikelihood


def runTestCases(cleanData, scaledFeatures, stateSequence, posteriors,
                 model, stateMap, scaler):
    print("\n" + "=" * 60)
    print("  TEST CASE ANALYSIS")
    print("=" * 60)

    WINDOW = 30
    dates = np.array(cleanData['date'].values)
    T = len(stateSequence)
    originalFeatures = scaler.inverse_transform(scaledFeatures)

    # Identify state index for Low and High regimes
    lowIdx = next(k for k, v in stateMap.items() if v == 'Low Volatility')
    highIdx = next(k for k, v in stateMap.items() if v == 'High Volatility')

    # Pre-compute per-window fractions for each regime
    lowFracs = np.array([
        np.mean(stateSequence[i:i + WINDOW] == lowIdx)
        for i in range(T - WINDOW)
    ])
    highFracs = np.array([
        np.mean(stateSequence[i:i + WINDOW] == highIdx)
        for i in range(T - WINDOW)
    ])
    balance = np.abs(lowFracs - highFracs)

    caseIndices = [
        ('Test Case 1: Stable Market Period',    int(np.argmax(lowFracs))),
        ('Test Case 2: Turbulent Market Period', int(np.argmax(highFracs))),
        ('Test Case 3: Regime Transition',       int(np.argmin(balance))),
    ]

    maxEntropy = np.log(model.n_components)  # theoretical maximum

    for caseName, startIdx in caseIndices:
        endIdx = min(startIdx + WINDOW, T)
        windowDates = dates[startIdx:endIdx]
        windowStates = stateSequence[startIdx:endIdx]
        windowPosts = posteriors[startIdx:endIdx]
        windowFeats = originalFeatures[startIdx:endIdx]

        # Shannon entropy of posterior as timestep-level uncertainty measure
        entropy = -np.sum(
            windowPosts * np.log(np.clip(windowPosts, 1e-10, 1)), axis=1
        )
        normEntropy = entropy.mean() / maxEntropy   # 0=certain, 1=maximally uncertain

        dominantState = stateMap[int(np.bincount(windowStates).argmax())]
        stateCounts = {
            stateMap[s]: int(np.sum(windowStates == s))
            for s in range(model.n_components)
        }
        meanPosteriors = {
            stateMap[i]: round(float(windowPosts.mean(axis=0)[i]), 4)
            for i in range(model.n_components)
        }

        print(f"\n  {caseName}")
        print(f"    Period             : {windowDates[0]} → {windowDates[-1]}")
        print(f"    Input (mean)       : Volatility={windowFeats[:, 0].mean():.2f}, "
              f"Cloud Cover={windowFeats[:, 1].mean():.2f}%")
        print(f"    Dominant State     : {dominantState}")
        print(f"    State Counts       : {stateCounts}")
        print(f"    Mean Posteriors    : {meanPosteriors}")
        print(f"    Mean Entropy       : {entropy.mean():.4f} nats "
              f"(Normalised {normEntropy:.3f}; 0=certain, 1=uncertain)")
        print(f"    First 10 States    : "
              f"{[stateMap[s] for s in windowStates[:10]]}")

    return caseIndices



# VISUALISATIONS
def visualiseModelSelection(resultsDF):
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(resultsDF['nStates'], resultsDF['bic'], 'b-o', label='BIC', linewidth=2)
    ax1.set_xlabel('Number of Hidden States')
    ax1.set_ylabel('BIC (lower is better)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.plot(resultsDF['nStates'], resultsDF['logLikelihood'],
             'r--s', label='Log-Likelihood', linewidth=2)
    ax2.set_ylabel('Log-Likelihood (higher is better)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    bestN = int(resultsDF.loc[resultsDF['bic'].idxmin(), 'nStates'])
    ax1.axvline(x=bestN, color='green', linestyle=':', linewidth=2,
                label=f'BIC Optimal N={bestN}')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title('HMM Model Order Selection: BIC vs Log-Likelihood')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}hmmModelSelection.png", dpi=150)
    plt.close()
    print("  Saved: hmmModelSelection.png")


def visualiseStateSequence(cleanData, stateSequence, stateMap, stateColourMap):
    dates = pd.to_datetime(cleanData['date'].values)
    volatility = cleanData['xejVolatility'].values

    fig, ax = plt.subplots(figsize=(16, 6))

    # Background shading for each state regime
    for state, label in stateMap.items():
        mask = stateSequence == state
        ax.scatter(dates[mask], volatility[mask],
                   c=stateColourMap[state], label=label,
                   alpha=0.65, s=8, zorder=3)

    ax.plot(dates, volatility, color='gray', alpha=0.2, linewidth=0.6, zorder=1)

    ax.set_title('XEJ Daily Volatility Coloured by HMM Hidden State (Viterbi Decoding)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Volatility (High − Low)')
    ax.legend(markerscale=3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}hmmStateSequence.png", dpi=150)
    plt.close()
    print("  Saved: hmmStateSequence.png")


def visualiseTransitionMatrix(model, stateMap):
    labels = [stateMap[i] for i in range(model.n_components)]

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        model.transmat_, annot=True, fmt='.4f', cmap='Blues',
        xticklabels=labels, yticklabels=labels,
        vmin=0, vmax=1, ax=ax, linewidths=0.5, linecolor='white'
    )
    ax.set_title('Learned HMM Transition Probability Matrix (A)')
    ax.set_xlabel('Destination State (t+1)')
    ax.set_ylabel('Source State (t)')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}hmmTransitionMatrix.png", dpi=150)
    plt.close()
    print("  Saved: hmmTransitionMatrix.png")


def visualiseEmissions(model, stateMap, stateColourMap, scaler):
    originalMeans = scaler.inverse_transform(model.means_)
    featureNames = ['XEJ Volatility (High − Low)', 'Mean Daily Cloud Cover (%)']
    xRanges = [
        np.linspace(0, originalMeans[:, 0].max() * 2.5, 400),
        np.linspace(0, 100, 400)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for featIdx, (ax, featName, xRange) in enumerate(zip(axes, featureNames, xRanges)):
        for stateIdx in range(model.n_components):
            mean = originalMeans[stateIdx, featIdx]
            # Marginal std from full covariance, unscaled back to original units
            scaledVar = model.covars_[stateIdx][featIdx, featIdx]
            originalStd = np.sqrt(scaledVar) * scaler.scale_[featIdx]

            pdf = spnorm.pdf(xRange, loc=mean, scale=originalStd)
            ax.plot(xRange, pdf,
                    color=stateColourMap[stateIdx],
                    label=f"{stateMap[stateIdx]} (μ={mean:.1f}, σ={originalStd:.1f})",
                    linewidth=2.5)
            ax.axvline(mean, color=stateColourMap[stateIdx],
                       linestyle='--', alpha=0.35, linewidth=1)

        ax.set_title(f'Emission Distributions: {featName}')
        ax.set_xlabel(featName)
        ax.set_ylabel('Probability Density')
        ax.legend(fontsize=8.5)

    plt.suptitle('Gaussian Emission Distributions per Hidden State', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}hmmEmissions.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: hmmEmissions.png")


def visualiseUncertainty(cleanData, posteriors, stateMap, stateColourMap):
    dates = pd.to_datetime(cleanData['date'].values)

    fig, ax = plt.subplots(figsize=(16, 5))
    bottom = np.zeros(len(dates))

    for state in sorted(stateMap.keys()):
        ax.fill_between(
            dates, bottom, bottom + posteriors[:, state],
            color=stateColourMap[state], alpha=0.75,
            label=stateMap[state]
        )
        bottom += posteriors[:, state]

    ax.set_ylim(0, 1)
    ax.set_title('State Posterior Probabilities Over Time (Forward-Backward Algorithm)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Posterior Probability P(state | all observations)')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}hmmUncertainty.png", dpi=150)
    plt.close()
    print("  Saved: hmmUncertainty.png")


def visualiseCloudVsState(cleanData, stateSequence, stateMap, stateColourMap):
    """Saves box plots of cloud cover distributions stratified by hidden state."""
    cleanData = cleanData.copy()
    cleanData['state'] = [stateMap[s] for s in stateSequence]

    stateOrder = ['Low Volatility', 'Normal Volatility', 'High Volatility']
    palette = {label: stateColourMap[idx]
               for idx, label in stateMap.items()}

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=cleanData, x='state', y='cloudCover',
        order=stateOrder, palette=palette, ax=ax, width=0.5
    )
    ax.set_title('Cloud Cover Distribution by HMM Hidden State')
    ax.set_xlabel('Hidden State (Volatility Regime)')
    ax.set_ylabel('Mean Daily Cloud Cover (%)')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}hmmCloudVsState.png", dpi=150)
    plt.close()
    print("  Saved: hmmCloudVsState.png")



# TEAM DATA CONTRACT
def exportDataContract(cleanData, stateSequence, posteriors, model, stateMap, scaler):
    """Assembles and exports the structured output that feeds the team DAG.

    This function defines the data contract for this module — the agreed schema
    by which downstream team modules consume my HMM outputs. It returns
    a dict and also persists a CSV so teammates can load the data without running
    the full pipeline.

    Data Contract Schema

    Per-day sequence output (DataFrame, one row per trading day):
        date                  : datetime.date — trading day
        xejVolatility         : float — raw intra-day range (High - Low)
        cloudCover            : float — mean daily cloud cover (%)
        viterbi_state_index   : int   — raw HMM state index (0, 1, 2)
        viterbi_state_label   : str   — 'Low Volatility' | 'Normal Volatility'
                                        | 'High Volatility'
        posterior_low         : float — P(Low Volatility  | all obs), [0, 1]
        posterior_normal      : float — P(Normal Volatility | all obs), [0, 1]
        posterior_high        : float — P(High Volatility  | all obs), [0, 1]
        state_entropy         : float — Shannon entropy of posterior (nats);
                                        0 = certain, log(N) = maximally uncertain

    Model-level outputs (dict keys):
        transition_matrix     : pd.DataFrame (N x N) of A[i,j] = P(j | i)
        emission_means        : pd.DataFrame — per-state feature means (original scale)
        model_log_likelihood  : float

    Args:
        cleanData: Cleaned DataFrame with raw feature columns.
        stateSequence: Viterbi-decoded state index array of shape (T,).
        posteriors: Forward-Backward posterior matrix of shape (T, N).
        model: Fitted GaussianHMM model.
        stateMap: Dict mapping state index -> label string.
        scaler: Fitted StandardScaler for inverse-transforming emission means.

    Returns:
        contract: dict containing 'sequence' (DataFrame) and model-level outputs.
    """
    # Build per-day sequence DataFrame 
    lowIdx    = next(k for k, v in stateMap.items() if v == 'Low Volatility')
    normIdx   = next(k for k, v in stateMap.items() if v == 'Normal Volatility')
    highIdx   = next(k for k, v in stateMap.items() if v == 'High Volatility')

    entropy = -np.sum(
        posteriors * np.log(np.clip(posteriors, 1e-10, 1)), axis=1
    )

    sequenceDF = cleanData[['date', 'xejVolatility', 'cloudCover']].copy()
    sequenceDF['viterbi_state_index'] = stateSequence
    sequenceDF['viterbi_state_label'] = [stateMap[s] for s in stateSequence]
    sequenceDF['posterior_low']       = posteriors[:, lowIdx]
    sequenceDF['posterior_normal']    = posteriors[:, normIdx]
    sequenceDF['posterior_high']      = posteriors[:, highIdx]
    sequenceDF['state_entropy']       = entropy

    # Transition matrix 
    labels = [stateMap[i] for i in range(model.n_components)]
    transitionDF = pd.DataFrame(
        model.transmat_,
        index=labels,
        columns=labels
    )

    # Emission means (original scale) 
    invMeans = scaler.inverse_transform(model.means_)
    emissionDF = pd.DataFrame(
        invMeans,
        index=labels,
        columns=['mean_xejVolatility', 'mean_cloudCover']
    )

    contract = {
        'sequence':            sequenceDF,
        'transition_matrix':   transitionDF,
        'emission_means':      emissionDF,
        'model_log_likelihood': model.score(
            scaler.transform(cleanData[['xejVolatility', 'cloudCover']].values)
        ),
    }

    # Persist to CSV for teammates who import without running the pipeline 
    contractPath = f"{OUTPUT_DIR}dataContract_wick0133.csv"
    sequenceDF.to_csv(contractPath, index=False)
    transitionDF.to_csv(f"{OUTPUT_DIR}dataContract_transitionMatrix_wick0133.csv")
    print(f"\n  [Data Contract] Sequence output  → {contractPath}")
    print(f"  [Data Contract] Transition matrix → "
          f"{OUTPUT_DIR}dataContract_transitionMatrix_wick0133.csv")

    return contract


def runPipeline():
    sns.set_theme(style='whitegrid')

    cleanData                        = preprocessData()
    scaledFeatures, scaler           = prepareFeatures(cleanData)
    model                            = trainHMM(scaledFeatures, nStates=FINAL_N_STATES)
    stateMap, stateColourMap, _      = labelStates(model, scaler)
    _, stateSequence                 = decodeWithViterbi(model, scaledFeatures)
    posteriors                       = computeForwardBackward(model, scaledFeatures)
    contract                         = exportDataContract(
                                           cleanData, stateSequence, posteriors,
                                           model, stateMap, scaler
                                       )
    return contract



# MAIN PIPELINE
def main():
    sns.set_theme(style='whitegrid')
    print("=" * 60)
    print("  SEQUENTIAL HMM PIPELINE — XEJ VOLATILITY REGIME DETECTION")
    print("=" * 60)

    # Data Ingestion 
    print("\n[1/6] Data Ingestion and Preprocessing")
    cleanData = preprocessData()
    print(f"  Dataset: {len(cleanData)} trading days "
          f"({cleanData['date'].iloc[0]} to {cleanData['date'].iloc[-1]})")
    print(f"  Missing values: {cleanData.isna().sum().sum()}")

    # Feature Preparation 
    print("\n[2/6] Feature Preparation (StandardScaler)")
    scaledFeatures, scaler = prepareFeatures(cleanData)
    print(f"  Feature matrix shape: {scaledFeatures.shape}  "
          f"[xejVolatility, cloudCover]")

    # Model Order Selection 
    print("\n[3/6] Model Order Selection (BIC)")
    resultsDF, bestN = selectModelOrder(scaledFeatures)
    print(f"\n  NOTE: Final model uses {FINAL_N_STATES} states per system design "
          f"(BIC suggested {bestN}).")

    # Training and Inference 
    print(f"\n[4/6] HMM Training and Inference")
    model = trainHMM(scaledFeatures, nStates=FINAL_N_STATES)
    stateMap, stateColourMap, _ = labelStates(model, scaler)

    print("\n  Labelled states (original-scale volatility means):")
    invMeans = scaler.inverse_transform(model.means_)
    for idx, label in stateMap.items():
        print(f"    State {idx} → {label:20s} | "
              f"Mean Vol={invMeans[idx, 0]:.2f}, "
              f"Mean Cloud={invMeans[idx, 1]:.2f}%")

    viterbiLogProb, stateSequence = decodeWithViterbi(model, scaledFeatures)
    posteriors = computeForwardBackward(model, scaledFeatures)

    # Evaluation
    print("\n[5/6] Model Evaluation")
    evaluateModel(model, scaledFeatures, stateSequence, stateMap, resultsDF)
    runTestCases(
        cleanData, scaledFeatures, stateSequence,
        posteriors, model, stateMap, scaler
    )

    # Visualisations + Data Contract Export
    print("\n[6/6] Generating Visualisations")
    visualiseModelSelection(resultsDF)
    visualiseStateSequence(cleanData, stateSequence, stateMap, stateColourMap)
    visualiseTransitionMatrix(model, stateMap)
    visualiseEmissions(model, stateMap, stateColourMap, scaler)
    visualiseUncertainty(cleanData, posteriors, stateMap, stateColourMap)
    visualiseCloudVsState(cleanData, stateSequence, stateMap, stateColourMap)

    exportDataContract(cleanData, stateSequence, posteriors, model, stateMap, scaler)

    print("\n" + "=" * 60)
    print(f"  Pipeline complete. Outputs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()