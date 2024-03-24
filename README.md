# ISS_Seminar_GNN_SOTA
_________________________________________________________________________________________
Infomations Systems Seminar Project: GNN's as State of the Art Recommender Systems

Autors: Ali Jaabous, Bora Eguz, Ozan Ayhan

Supervisor: 

This repository will contain the necessary scripts and jupyter notebooks to reproduce our project proceedings

# Intro

__________________________________________________________________________________________

hen people want to listen to different music or watch different movies, it is not hidden that
they will seek the ones with similar tastes by hoping to like them. In this point,
recommendation systems can have a big effect on people’s decisions on whether to use the
same platform or continue their research on other platforms. Recent years showed that there
are fast and big developments in the Graph Learning based Recommender Systems (GLRS)
which enables us to explore homogeneous or heterogeneous relations in graphs [5]. The basic
idea of GNN is to model complex relationships between entities in a network. In the context
of recommendation systems, this means that not only the isolated preferences of a user or the
characteristics of a product are taken into account, but also the relationships between users
and products in a network [7].
Our goal in the scope of the seminar course is to build recommendation models with different
algorithms to compare them. To do this, the Netflix data (which can be found on Kaggle) will
be manipulated (also will be downsized for the prototype). This data is from a competition
Netflix organized back in 2009 to have the best recommendation system. They provided the
data of movie information and user’s rating. But as mentioned above, this time our goal is not
to find the best model but rather to benchmark GNN models and traditional recommendation
system models to see if it is the state-of-the-art model. To do this, we will apply some
preprocess steps to data, build and train the model, and lastly evaluate its performance with
some common metrics such as MAE or RMSE and also compare their execution times.
In the next section you will find the literature review focusing on the papers that have used
graph neural networks in their recommendation systems and also other models that focuses
on recommendation systems. After that, there is the methodology part which explains the
steps and algorithms that are used. Experimental design, the part 4, demonstrates all the data
preprocessing and feature engineering steps. And then explains the data splitting strategy and
fine tuning. In the same section model architecture and evaluation steps are being explained.
The results section shows all the metrics for each model and explains it verbally. At last, the
1
conclusion section is the part where the outcomes of this paper is being discussed.

![image](https://github.com/ISSeminarGNNSOTA/ISS_Seminar_GNN_SOTA/assets/162732442/bf387468-62ab-4ebc-a892-901f0dcfe791)

