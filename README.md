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

## For a more configured registration of CBCT and CT scans: 

### Prerequisites:
SimpleElastix build of SimpleITK: This can be obtained as an egg file from the repo

Run the following
```
python tools/register_scans.py --help
```

## General Info

The repository contains two modules
- dicom_saver
- registration_tools


