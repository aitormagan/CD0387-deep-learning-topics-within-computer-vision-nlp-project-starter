# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Files

* `hpo.py`: Used for hyperparameter tuning
* `train_model.py`: Used for debugging and profiling. It extends the `hpo.py` file and and the required hooks. In addition, it also logs loss.
* `inference.py`: It includes the required methods (`model_fn` to load the model and `input_fn` to transform the input into something which can be understood by the model) for the model to be deployed.  

## Hyperparameter Tuning

The model which was used for this project was ResNet18 as is the one which is prepared to deal with image identification. 

Here is the list of hyperparameters which have been tuned. They have been chosen according to the instructions given throughout the course and AWS recommendations:

* `lr`
* `epochs`
* `batch-size`

Here is a screen shot with some completed training jobs:

![Endpoint Running](./screenshots/completed_training_jobs.png)

Attached, you can also find a screenshot where you can see all the hyperparameter tuning jobs with their score:

![Endpoint Running](./screenshots/hyperparameter_tuning.png)

The best model we obtain was the one with these hyperparameters:

* `lr`: 0.0018658613084312835
* `epochs`: 32
* `batch-size`: 4


## Debugging and Profiling

Debugging and profiling was performed according to the instructions given throughout the course. In this case, the following rules were applied:

```python
    ProfilerRule.sagemaker(rule_configs.LowGPUUtilization()),
    ProfilerRule.sagemaker(rule_configs.ProfilerReport()),
    Rule.sagemaker(rule_configs.vanishing_gradient()),
    Rule.sagemaker(rule_configs.overfit()),
```

In order to perform a correct debugging, the `train_model.py` has been created based on the `hpo.py` and setting up the required hooks before traning and testing.

### Results

The results of the debugging/profiling session arose the following output:

```
VanishingGradient: NoIssuesFound
Overfit: NoIssuesFound
LowGPUUtilization: NoIssuesFound
ProfilerReport: NoIssuesFound
```

As can be seen, no issues were found with this model.

In case issues were found, we will have to take a look at the documentation and check for possible solutions. 

## Model Deployment
To deploy the model, it has been required to create an extra file called `inference.py` which loads the model and transforms the input. 

To call the model, you just have to execute the following lines of code replacing `IMAGE_PATH` by the path where your image is stored and `ENDPOINT` by the name of your endpoint:

```python
import io
import sagemaker
from PIL import Image
from sagemaker.serializers import IdentitySerializer
from sagemaker.pytorch.model import PyTorchPredictor

serializer = IdentitySerializer("image/jpeg")
predictor = PyTorchPredictor(ENDPOINT, serializer=serializer, sagemaker_session=sagemaker.Session())

buffer = io.BytesIO()
Image.open(IMAGE_PATH).save(buffer, format="JPEG")
response = predictor.predict(buffer.getvalue())
```

Here is an image of the endpoint up and running:

![Endpoint Running](./screenshots/endpoint_running.png)
