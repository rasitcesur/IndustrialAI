param(
    [string]$DataDir = 'DataSet/casting',
    [ValidateSet('auto','cuda','cython','python')]
    [string]$Backend = 'auto'
)
$ErrorActionPreference = 'Stop'
python -m pip install -r requirements.txt
python main.py train --data-dir $DataDir --backend $Backend --model models/cv_atlas_mil.joblib
