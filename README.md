# Supervised Model for Predicting Default of Credit Card Clients
The goal of this experiment is to build a Multi-Layer Perceptron (MLP), a Deep Neural Network (DNN) model to predict the deafault of credit card clients.

**Keywords:** DNN, MLP, Supervised model, Prediction

## The Data
We used the [default of credit card clients](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) dataset, a case of customers default payment in Taiwan, from [UCI](https://archive.ics.uci.edu/ml/index.php). It has 24 attributes (number of columns) and 30000 instances (number of rows).

## Requirements
You can find the modules and libraries used in this project in the [requirement.txt](https://github.com/gloryodeyemi/COMP_8730_Project/blob/main/requirements.txt) file. You can also run the code below.
```
pip install -r requirements.txt
```

## Structure
* **[Data](https://github.com/gloryodeyemi/COMP_8730_Project/tree/main/Data):** contains the data file used for this project.

* **[utils](https://github.com/gloryodeyemi/COMP_8730_Project/tree/main/utils):** contains the essential functions used for the data analysis.

* **[data_analysis.ipynb](https://github.com/gloryodeyemi/COMP_8730_Project/blob/main/data_analysis.ipynb):** A python notebook that uses the function in the utils to analyse the data used in this project. The results gives information about the data.

* **[data_collection.ipynb](https://github.com/gloryodeyemi/COMP_8730_Project/blob/main/data_collection.ipynb):** A python notebook that shows you the procedure of collecting tweets from Twitter using the Twitter API and tweepy python library.

* **[quick_start.ipynb](https://github.com/gloryodeyemi/COMP_8730_Project/blob/main/quick_start.ipynb):** A python notebook that shows a successful run of the project using the quickstart guideline.

* **[Summarization.ipynb](https://github.com/gloryodeyemi/COMP_8730_Project/blob/main/Summarization.ipynb) and [Summarization.py](https://github.com/gloryodeyemi/COMP_8730_Project/blob/main/Summarization.py)** are python notebook and script that shows the procedure of summarizing tweets with English-Yoruba code switches and the result gotten.

## Quickstart Guideline
1. Clone the repository
``` 
git clone https://github.com/gloryodeyemi/COMP_8730_Project.git 
```
2. Change the directory to the cloned repository folder
```
%cd .../COMP_8730_Project
```
3. Install the needed packages
```
pip install -r requirements.txt
```
4. Run the script
```
python Summarization.py
```

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/gloryodeyemi/COMP_8730_Project/blob/main/LICENSE) file for details.

## Contact
Glory Odeyemi is currently undergoing her Master's program in Computer Science, Artificial Intelligence specialization at the [University of Windsor](https://www.uwindsor.ca/), Windsor, ON, Canada. You can connect with her on [LinkedIn](https://www.linkedin.com/in/glory-odeyemi-a3a680169/).

## References
1. [Tweepy](https://www.tweepy.org/)
