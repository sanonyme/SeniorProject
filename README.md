# 📚 Project description
As part of my computer science senior project, I developed a set of machine learning algorithms that would predicted ideal crops based on soil data. These models are to be used inside a web app, through a communication of a RESTful API made via the backend.
# 🏃‍♂️ Running the project
__Steps:__

1) Install python from this link: https://www.python.org/downloads/
2) Library installastions needed for running the app:
```
pip install Flask
```
```
pip install numpy
```
> **_NOTE:_**  Use sci-kit learn 1.4.2 version

```
pip install -U scikit-learn
```
```
pip install scikit-learn==1.4.2
```


## 🔧 Tools used
1) Pandas for data retriveal and exploration
2) Numpy for data manipulation 
3) Sci-kit learn for model development
4) Flask to make a production ready application for crop predictions as a RESTful API, used in an web application.

# 🖥 Model rundown
This project used 5 classification algorithms to achieve the desired predictions, which are:
- KNN
- Decision tree
- SVC (Linear, Poly, RBF)
- Random forest
- XGBoost

As a result of these algorithms an average predictions of __97%__ was achieved.

To further improve the performance of the model, soft voting ensembling was used, which further increased the models performance with a __99%__ accuracy.

# 🧠 Creating predictions
1) Crop predictions can be made in 2 ways, the first is through a generic HTML interface (No designs were made as the purpose is to use this app as an API)

![1](https://github.com/Kassem-Faraj/SeniorProject/assets/67020401/13f93401-0aa2-40e0-ad18-1e4a93e46074)


2) The second way (The intended use), is through a JSON input through the /prediction endpoint
