# 4170-Project Group 13

Our implementation consists of 6 files 
 - `Webscraper.py` - Python file the scrapes https://www.eliteprospects.com to colect data on all players who have played in the NHL
 - `clean_data.py` - Python file that performs preliminary cleaning to the data. ie remove seasons without any games player, remove seaosn played in invalid leagues and remove players who have not played 30 nhl games. 
 - `feature_engineering.py` - Python file that creates per-game rate features and normalizes games played.
 - `train.py` - Jupyter notebook which has some additonal feature engineering, then it creates our model, trains our model and evaluate's our model. 
 - `neural.py` - Python files containing our custom torch.nn module for our MLP model. It defining the structure of the model to be used in `train.py`.
 - `dataset.py` - Python files containing a custom torch dataset which we use for feeding input value and output labels to our in `train.py`.

To run our python files you will need to have the following python libraries installed.
- torch - install using `pip install torch torchvision torchaudio`
- matplotlib - install using `pip install matplotlib`
- sklearn - install using `pip install scikit-learn`
- httpx - install using `pip install httpx`
- pathlib (if python version before python 3.4) - install using `pip install pathlib`


To execute our code follow these steps

### **1** 
First you run `Webscraper.py` to collect the data which can be done by running the following in your command line
```
python Webscraper.py
```
or
```
python3 Webscraper.py
```
in your command line. 

Note this will take a long time to exewcute due to the slow response speed of https://www.eliteprospects.com and the need to send a seperate request for each player. THe program will print done to the command line when it is finished. 

### **2** 
Once `Webscraper.py` is done running you can next run `clean_data.py`. When running `clean_data.py` there are optional command line arguments to change the file name of the input data for the script and to change the output file name. By default it takes input from `playerData.txt`(the file Webscraper outputs to) and outputs the cleaned data to `playerData_cleaned.txt`. It can be run as follows in the command line. 
```
python clean_data.py
```
or
```
python3 clean_data.py
```
or
```
python clean_data.py inputfilename outputfilename
```
or
```
python3 clean_data.py inputfilename outputfilename
```

### **3** 
Next you can run `feature_engineering.py` which is run in same fashion as `clean_data.py` was with optional command line arguments to change the default input file name to no longer be `playerData_cleaned.txt` and the default output file name to no longer be `playerData_features.txt`. It can be run as follows in the command line.
```
python feature_engineering.py
```
or
```
python3 feature_engineering.py
```
or
```
python feature_engineering.py inputfilename outputfilename
```
or
```
python3 feature_engineering.py inputfilename outputfilename
```

### **4**
Finally you can run `train.py` as follows in the command line
```
python train.py
```
or
```
python3 train.py
```
