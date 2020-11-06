# Streamlit Machine Learning

The dataset that is used is from Kaggle.com (https://www.kaggle.com/c/dog-breed-identification/overview). Each image has his own unique ID with the corresponding dog breed. It consists of a collection of 10,000+ labelled images of 120 different dog breeds.

Who's a good dog? Who likes ear scratches? Well, it seems those fancy deep neural networks don't have all the answers. However, maybe they can answer that ubiquitous question we all ask when meeting a four-legged stranger: what kind of good pup is that?

The original ImageNet set has quite a few different dog classes so we can reuse CNNs with pretrained ImageNet weights. With that model as a base layer, we applied some Transfer Learning by added a new layer and make this CNN learn different dog breeds on this dataset.
