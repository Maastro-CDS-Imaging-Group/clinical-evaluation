# Repository for clinical evaluation 

## Installation

```
python setup.py install
```


Once installed, the scripts can be run using
```
save_dicom -h # Check the parameters required
deform -h # Check the parameters required
```


# Projects

## CBCT-CT validation preparation
### Prerequisites:
SimpleElastix build of SimpleITK: This can be obtained as an egg file from the repo

Run the following for photon dataset, 
```
python tools/register_photon_scans.py --help
```

Run the following for proton dataset, 
```
python tools/register_proton_scans.py --help
```
## General Info

The repository contains two modules
- dicom_saver
- registration_tools


