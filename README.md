# Multimodal fused deep learning models for drug molecular property prediction

The biological information itself contains high-dimensional and complex information. Extracting the necessary information from biological data poses a challenging problem. We have developed a multimodal model that encodes biological molecules into sequence information, binary fingerprints, and graphs. By utilizing three different models for learning, we combine the learned features from each modality to achieve predictions of biological molecule properties.

## Multimodal deep learning offers several advantages for drug property prediction:

**Comprehensive Representation**: By incorporating multiple modalities such as molecular structures, chemical fingerprints, and biological sequences, multimodal deep learning provides a more comprehensive representation of the underlying data. This allows for a more holistic understanding of the complex relationships between various molecular features and drug properties.

**Enhanced Feature Learning**: Each modality contains unique and complementary information. By leveraging multiple modalities, multimodal deep learning models can learn rich and diverse features from each modality, capturing both local and global patterns. This leads to improved representation learning and enhanced predictive performance.

**Improved Prediction Accuracy**: Fusing information from different modalities can lead to improved prediction accuracy. By combining the strengths of each modality, multimodal deep learning can capture complex interactions and correlations that may not be apparent when using a single modality alone. This enables more accurate and reliable predictions of drug properties.

**Robustness to Missing Data**: In drug discovery and development, it is common to have missing data for certain modalities. Multimodal deep learning models can handle missing data more effectively by leveraging available modalities to make predictions. This robustness to missing data ensures that the models can still provide meaningful predictions even when some modalities are incomplete or unavailable.

Overall, multimodal deep learning approaches offer a powerful framework for drug property prediction, leveraging diverse sources of information to enhance predictive accuracy, robustness, and interoperability.

## Multimodal fused deep learning model
In this contribution, we conduct three studies
__Examine performance accuracy__: We validate the Multimodal fused deep learning (MMFDL) model on six single-molecule datasets.
__Identify the proper fusion strategy__: The learned feature needs to be combined to get the output. Not all the modal models will equally contribute to the output. Therefore, we need to identify the optimal approaches.
__Robustness to data noise__: We randomly add noise to the input data and compare the performance of single-modal and multimodal models.

## The input representation of multimodal fusion deep learning model

Molecular representations of molecules into SMILES-encoded vectors, ECFP and molecular graph.

**SMILES-encoded vectors**: For different datasets, a unified regular expression is used to construct a label dictionary.  The SMILES in the dataset are mapped to fixed-length integer sequences by the labeling dictionary. Every token in the SMILES sequence is subjected to encoding, followed by the generation of an embedding vector for each token. Any shorter SMILES strings were padded with zeros at the end.

**ECFP**: ECFP describes a complex as a fixed-length bit string, with each bit indicating the presence or absence of a particular substructure in the complex.

**Graph**: Molecular graphs include adjacency matrices and feature vectors. Each node represented by a feature vector composed of 78 values, including 44 atomic types, 11 one-hot encoding atom degree, 11 one-hot encoding of the total number of hydrogens, 11 one-hot encoding valences and 1 bit for the aromaticity. 

![Alt Text](https://github.com/AIMedDrug/MMFDL/raw/03c39afa047d7a2cb6f1e67535ad0320f68651b4/notebook/inputRepresentation.png)

## The structure representation of multimodal fusion deep learning model

The below figure is the structure of the multimodal fused deep learning (MMFDL) model. Blue, orange and green color lines represent the data flow in the model for training, tuning, and test sets. SMILES-encoded vectors are processed by Transformer-Encoder, ECFP is processed by BiGRU with MultiHead Attention, and Graph is processed by GCN model. The training set is used to train the feature extraction model; the validation set determines the hyperparameters of the feature extraction model. The tuning set is used to assign the weights for each modal input. The test set is used to validate the prediction performance.

![Alt Text](https://github.com/AIMedDrug/MMFDL/raw/03c39afa047d7a2cb6f1e67535ad0320f68651b4/notebook/MMFDL_model.png)

## Multimodal fused deep learning protocol 

The MMFDL needs two steps: (1) Generate input using three modal types as SMILES-encoded token, bit fingerprint and molecular graph; and split the data into training, tuning and test sets. (2) Train the model using Transformer, BiGRU, GCN or the fused triple-modal models.

We provide scripts for data processing and model training. You can use the .py and .ipynb if you are interested in MMFDL.

We have taken the SAMPL dataSet as an example, uploaded the source code to github, and can use google colab for testing.

**1 github**

github address: https://github.com/AIMedDrug/MMFDL.git

**2 Google colab's .ipynb file address for testing**

(1) Data processing
MMFDL_geneInput.ipynb: https://colab.research.google.com/drive/1hLjDxffFA0SN_bZaWzul2TW-zhkoJDov?usp=sharing

(2) Training model using unimodal and multimodal deep learning model
(2.1) Transformer model
MMFDL_singleTrans.ipynb: https://colab.research.google.com/drive/1JLzGPrSqOUKUYl9B2PkqhsussCtx4OvL?usp=sharing

(2.2) BiGRU
MMFDL_singleBiGRU.ipynb: https://colab.research.google.com/drive/1JLzGPrSqOUKUYl9B2PkqhsussCtx4OvL?usp=sharing

(2.3) GCN
MMFDL_singleGCN.ipynb: https://colab.research.google.com/drive/1RgNsyJ5PgmQtoa93nDkK5BT0luyCcgef?usp=sharing

(2.4) MMFDL
MMFDL_ML.ipynb: https://colab.research.google.com/drive/1QQIeOwU9jeuTmIyppQ5IvPYqOcHBarWp?usp=sharing

(2.5) MMFDL
MMFDL_SGD.ipynb: https://colab.research.google.com/drive/1phj6l-JViYmFBsPQ23I7o4T5JB9bH3ab?usp=sharing



