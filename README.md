# Palimpsest Text Contrast
An implementation of the Generalized Orthoginal Subspace Projection (GOSP) target detection method on multi-spectral, historical document imaging.


## Setup
(1) Clone the repository to your local machine using a terminal:

git clone https://github.com/AntiPersnlMyne/RIT_MISHA_GOSP


(2) Navigate to the project directory:

cd ~/PATH TO PROJECT/RIT_MISHA_GOSP


(3) Run the setup script:

(Linux) \
`bash setup.sh`

(Windows) \
`setup.bat`

(4) Move data to be processed into `data/input`


This script will:
- Install the required dependencies (libraries) from the `requirements.txt` file
- Create the necessary directory structure
- Optionally creates virtual environment - keeps dependencies isolated
- Delete the `setup.bat/.sh` file
--------------------------------------------------------------------------------



## Dependencies
This project uses the ENVI (Environment for Visualizing Imagery) software by NV5 Geospatial Software. This project is compatable with the now latest version of ENVI - 6.1 with IDL 9.1. Compatible Python versions are > 3.10.x and < 3.12.x. 

The software can be made available through a CIS (Chester F. Carlson College of Imaging Science) license. The MISHA (Multispectral Imaging System for Historical Artifacts) project is a system created by CIS and the RIT Museum Studies Program.

Python Version:
- 3.12.7 (This is due to a compatability issue with IDL 9.1)
- When prompted during startup.sh/.bat, creating the virtual environment will provide a compatable Python version for the project

This project includes the following libraries: 
1. Numpy (data arrays)
2. OpenCV (computer vision)
3. SPy (spectral data processing)
4. scikit-image (image processing)
5. SciPy (scientific computing)
6. Matplotlib (plotting)
7. Pytesseract (OCR)
8. Pillow/PIL (OCR dependancy)

The startup script AUTOMATICALLY downloads these libraries, ready to use, no user input required (venv recommended for package isolation)



## Usage
The current Python script is setup to accept TIFF (`.tif`, `.tiff`) raw image files, as well as ENVI files (`.pro`, `.dat`). 

Place all PRE-processed images into the input directory:

`data/input`

Images that have been POST-processed are stored in the output directory: 

`data/output`

Utility Python and IDL files are stored in their respective directories: 

`src/python_scripts`
`src/IDL_scripts`

The execution file is `main.py`


## Questions (or complaints)
Complaints may fall on deaf ears. Questions may fall on ignorant ears.

For questions concerning legality or contact from within CIS, please contact Gian-Mateo (AntiPersnlMyne) at: 
`mt9485@rit.edu`

### Additional Personel

Douglas Tavolette(dft8437@rit.edu) | RIT software engineering student, who devoted his free time to developing code for this project


Roger Easton Jr.(rlepci@rit.edu) | My excellent adviser and project sponsor, MISHA personell


David Messinger(dwmpci@rit.edu) | Another adviser and mentor, MISHA personell



