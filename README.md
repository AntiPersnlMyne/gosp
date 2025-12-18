# Generalized Orthogonal Subspace Projection (GOSP)
An implementation of the Generalized Orthoginal Subspace Projection (GOSP) by Chang & Ren (2000). Created for the purpose of my senior research capstone. 


## Setup
(1) Clone the repository to your local machine using a terminal:

git clone https://github.com/AntiPersnlMyne/gosp


(2) Navigate to the project directory:

cd gosp


(3) Run the setup script:

(Linux) \
`bash setup.sh`

(Windows) \
`setup.bat`


This script will:
- Install the required dependencies (libraries) from the `requirements.txt` file
- Delete the `setup.bat/.sh` file
--------------------------------------------------------------------------------



## Dependencies

Python Version:
- 3.13.x (3.14+ is untested as of writing)

This project includes the following libraries: 
1. Numpy 
2. rasterio
3. SPy
4. scikit-image 
5. SciPy 
6. Matplotlib 


## Usage
The current Python script is setup to accept TIFF (`.tif`, `.tiff`) raw image files, as well as NumPy datafiles (.npy)

Test files are stored in `/tests`

The main execution file is `./main.py`


## Questions (or complaints)
Complaints may fall on deaf ears. Questions may fall on ignorant ears.

Contact Gian-Mateo (AntiPersnlMyne) at: 
`mt9485@rit.edu`

