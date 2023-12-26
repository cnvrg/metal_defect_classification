# Metal Defect Classification

Use this blueprint with your custom data to train a tailored model and deploy an endpoint that classify metal defects in images. Training a metal defect classifier algorithm requires data provided in the form of metal defect-containing images and their labels.

Complete the following steps to train this metel defect detector model:
1. Click the **Use Blueprint** button. The cnvrg Blueprint Flow page displays.
2. In the flow, click the **S3 Connector** task to display its dialog.
   * Within the **Parameters** tab, provide the following Key-Value pair information:
     - **Key**: `bucketname` - **Value**: enter the data bucket name
     - **Key**: `prefix` - **Value**: provide the main path where the data folder is located
   * Click the **Advanced** tab to change resources to run the blueprints, as required.
3. Return to the flow and click the **Metal Defect** task to display its dialog.
   * Within the **Parameters** tab, provide the following Key-Value pair information:
     - **Key**: `train_path` – **Value**: provide the path to the images including the S3 prefix, with the following format: `/input/s3_connector/<prefix>/train`
     - **Key**: `test_path` – **Value**: provide the path to the labels including the S3 prefix, with the following format: `/input/s3_connector/<prefix>/validation`
   NOTE: You can use prebuilt example data paths already provided.
   * Click the **Advanced** tab to change resources to run the blueprints, as required.

7. Click the **Serving** tab in the project and locate your endpoint. Complete one or both of the following options:
   * Use the Try it Live section with any metal image to check the model.
   * Use the bottom Integration panel to integrate your API with your code by copying in your code snippet.
   
A custom model that can classify metal defect in images has now been trained and deployed.
[See here how we created this blueprint](https://github.com/cnvrg/metal_defect_classification)