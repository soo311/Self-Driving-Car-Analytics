# Self-Driving-Car-Analytics
Code Challenge for Self Driving Car DS

To run the script, please follow the following steps.
1. Unzip the "data.zip" file
2. Place the folder into the directory where you want to run. For me, I placed the folder in "/orbital/soo/"
3. In order to run the script, you need to specify the two paths. For me, I specified them in the script itself, but you could pass in as an argument using the argparse.
  * LOG_DIR = '/orbital/soo/data/log' (Log file will be saved here)
  * DATA_DIR = "/orbital/soo/data/"
4. Add the python path: "export PYTHONPATH=$PYTHONPATH:/orbital/soo/data"
5. Example command line in Terminal: python run.py -f '/orbital/soo/data/'
6. There is an initial EDA for the analysis in the Data folder. 
