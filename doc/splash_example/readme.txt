In this folder is a script, gen_models.py, that generates the SplashRNA model and uses this model to predict on the miR-E training data. The script reads in the mir30_data.txt and mirE_data.txt files and generates two SVM models: mir30_model.bz2 and mirE_model.bz2. Finally, these two SVMs are combined to create the final cascaded SplashRNA classifier. This classifier is then used to predict on the miR-E training data, mirE_data.txt. 

gen_models.py requires Shogun in order to run. Shogun can be downloaded from here: http://shogun-toolbox.org/install 

The miR-E model provided in this directory (mirE_model.bz2) was trained using slightly more data than is provided in the example directory. The provided model is the miR-E component of the classifier used for all published predictions and on splashrna.mskcc.org. The model generated using the example training data will differ slightly from the provided model.
