20/11/2025 - Castro

1.  Project Overview

This project implements a two-stage hybrid acoustic calibration workflow for the Classroom_OBJ model using the Treble SDK. The goal is to calibrate material absorption coefficients ($\alpha$) and scattering coefficients ($\sigma$) to minimize the error between simulated reverberation time ($T_{20}$) and measured ground truth data.The project utilizes a Genetic Algorithm (GA) for broad search (Stage A) followed by an optional local search (Stage B or Hill-Climbing) for fine-tuning.
Model: classroom1Target 
Metric: Reverberation Time (T20) Simulation 
Type: Hybrid Acoustics (Wave-based below 720 Hz, Geometric above 720 Hz)
Optimization Range: 250Hz - 8000Hz
(GA Search)Targeted Low-Frequency Bands: 63Hz-125Hz (for post-GA fine-tuning, not implemented yet)

2. ⚙️ Repository Structure

This repository contains the core scripts and configuration files necessary to run the calibration workflow..
├── configs/
│   └── project.yaml              # MASTER CONFIG: Defines model, paths, bands, and GA/DG settings.
├── src/
│   ├── run_stage1_and_forward.py # Script for a single simulation run (used as the GA evaluation function).
│   ├── run_stage3_ga.py          # Main driver for the two-stage Genetic Algorithm calibration.
│   └── postprocess_stageA_v3.py  # Script to compile GA results (League Table, Error, Alpha analysis).
├── data/
│   └── omni/targets_measured_omni.json # Ground truth measured data (Reverberation Time).
├── input/
│   ├── directivity		  # Genelec directivity GenelecOy-8030_v2.cf2
│   └── obj 			  # 3D models
├── README.md
└── scripts/
    ├── *.*                       # various helper functions
    └── ss/ 			  # superseeded folder 

3.  Setup and Dependencies

This project requires the Treble SDK and standard Python data science libraries. Clone the Repository

Install Dependencies:It is recommended to use a virtual environment. 
	install treble-tsdk pandas numpy pyyaml
	# Note: Treble TSDK version should match the one used during development.

Configure Paths:Ensure the file paths within configs/project.yaml are correctly set for your local environment (especially model_obj, ground_truth_json, and working_dir).

4. Execution 

WorkflowThe calibration process is executed in four main steps: 
	4.1 Statistical & Forward (seed genearation) 
	4.2 GA Search, 
	4.3 Post-Processing, and 
	4.4 Low-Frequency Fine-Tuning. >>>>>> Not Implemented yet <<<<<<< 

Run command 

Stage 1 and forward:
python -m src.run_stage1_and_forward --config .\configs\project.yaml --run_label seed_run_006

Stage 3 GA
python .\src\run_stage3_ga.py --config .\configs\project.yaml --workdir .\results\seed_run_006_20251119_174112 --label GA_calib_008

Stage 3 GA analysis
python .\src\postprocess_stageA_v3.py --pattern "_calib_008_A_" --input_dir "C:\Users\usuario\Documents\TrebleSDK\v2\results" --project_config "C:\Users\usuario\Documents\TrebleSDK\v2\configs\project.yaml" --stage1_best "C:\Users\usuario\Documents\TrebleSDK\v2\results\seed_run_006_20251119_174112\best_individual_stageA.json" --output_excel "C:\Users\usuario\Documents\TrebleSDK\v2\results\postprocess_calib008A.xlsx"

TO DO
Low frequency analysis
Scattering


Output: An Excel file (Analysis_Report_GA_calib_008_A.xlsx) containing the League Table of all runs, error summaries, and the optimal absorption coefficients.C. Low-Frequency Fine-Tuning (Hill-Climbing)Since the GA was configured to ignore low-frequency loss (below $250 \text{ Hz}$), a dedicated local search is required for $63 \text{ Hz}$ and $125 \text{ Hz}$.Start Point: Load best_individual_stageA.json.Lock Values: Manually or programmatically lock all $\alpha$ values from $250 \text{ Hz}$ and up to prevent changing the mid/high-frequency match.Iterative Search: Implement a separate Hill-Climbing script (or modify a run script) to iteratively test small, $\pm 5-10\%$ changes to the 63 Hz and 125 Hz $\alpha$ values for the largest surface materials (e.g., My_Linoleum_on_Slab, walls, ceiling) and choose the option with the lowest $63 \text{ Hz}/125 \text{ Hz}$ error.5. 🔑 Key Parameters (project.yaml excerpts)ParameterLocationDefault ValueNotesBands for Optimizationbands:optimise_from_hz250Defines the threshold where loss is calculated. Kept at 250 Hz to manage low-frequency DG complexity during the main GA.Hybrid Crossovercalculation:crossover_frequency_hz720Separates the Wave-based (DG) and Geometric (GA) solvers. The DG solver handles the $63 \text{ Hz}$ and $125 \text{ Hz}$ bands.GA Populationga:pop_size7Number of candidate solutions per generation.GA Generationsga:generations_A5Number of evolutionary steps for the absorption $\alpha$ calibration. Recommended to increase to 25–30 for full convergence.