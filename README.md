# The Eye in the Sky
This is the repository for the problem statement *The Eye in the Sky* for the Inter IIT Tech Meet 2018 at IIT Bombay.
The approach used is: [Interactive Medical Image Segmentation using Deep Learning with Image-specific Fine-tuning](http://discovery.ucl.ac.uk/10032237/7/David_08270673.pdf) by Wang et al.

## Pre-Requisites
1. Download the dataset.
2. Install the python packages:
  ```
  pip install -r requirements.txt
  ```
3. Run the command:
  ```
  ./dataset_rot.py --data-path /path/to/train/dataset
  ```

## Instructions
### Retraining the best model
```
./best_model.py --data-path /path/to/train/dataset --save-dir /path/where/model/is/to/be/saved
```

### Reproducing the best model results
```
./evaluate.py --data-path /path/to/train/dataset --save-dir /path/where/model/is/saved
```
**NOTE**: The evaluation is done by excluding those pixels which are white in the target segmentation.
In order to include those pixels, use the `--include-white` argument.

### Getting predictions for test data
```
./inference.py --data-path /path/to/test/dataset --save-dir /path/where/model/is/saved
```

## Results
The following results were obtained by keeping "8.tif", "11.tif" and "12.tif" as test data, and training on the remaining images from the training dataset.

### Excluding White pixels
* Accuracy: 92.45%
* Cohen's Kappa: 0.9026
* Confusion Matrix:

<table align="center">
  <tr>
    <td align="center">3177778</td>
    <td align="center">674</td>
    <td align="center">3042</td>
    <td align="center">1798</td>
    <td align="center">26138</td>
    <td align="center">716</td>
    <td align="center">0</td>
    <td align="center">525</td>
    <td align="center">260984</td>
  </tr>
  <tr>
    <td align="center">725</td>
    <td align="center">1114190</td>
    <td align="center">89</td>
    <td align="center">0</td>
    <td align="center">59</td>
    <td align="center">0</td>
    <td align="center">0</td>
    <td align="center">2</td>
    <td align="center">10788</td>
  </tr>
  <tr>
    <td align="center">2297</td>
    <td align="center">219</td>
    <td align="center">1610603</td>
    <td align="center">6947</td>
    <td align="center">532</td>
    <td align="center">4</td>
    <td align="center">0</td>
    <td align="center">68</td>
    <td align="center">245576</td>
  </tr>
  <tr>
    <td align="center">971</td>
    <td align="center">0</td>
    <td align="center">5887</td>
    <td align="center">1626222</td>
    <td align="center">338</td>
    <td align="center">300</td>
    <td align="center">0</td>
    <td align="center">1</td>
    <td align="center">84661</td>
  </tr>
  <tr>
    <td align="center">25642</td>
    <td align="center">10</td>
    <td align="center">1039</td>
    <td align="center">439</td>
    <td align="center">4823761</td>
    <td align="center">156</td>
    <td align="center">1</td>
    <td align="center">2844</td>
    <td align="center">346080</td>
  </tr>
  <tr>
    <td align="center">659</td>
    <td align="center">0</td>
    <td align="center">124</td>
    <td align="center">1316</td>
    <td align="center">329</td>
    <td align="center">194079</td>
    <td align="center">0</td>
    <td align="center">34</td>
    <td align="center">5565</td>
  </tr>
  <tr>
    <td align="center">0</td>
    <td align="center">0</td>
    <td align="center">0</td>
    <td align="center">34</td>
    <td align="center">83</td>
    <td align="center">0</td>
    <td align="center">26891</td>
    <td align="center">0</td>
    <td align="center">1987</td>
  </tr>
  <tr>
    <td align="center">1220</td>
    <td align="center">0</td>
    <td align="center">41</td>
    <td align="center">0</td>
    <td align="center">1306</td>
    <td align="center">48</td>
    <td align="center">0</td>
    <td align="center">311439</td>
    <td align="center">10363</td>
  </tr>
  <tr>
    <td align="center">0</td>
    <td align="center">0</td>
    <td align="center">0</td>
    <td align="center">0</td>
    <td align="center">0</td>
    <td align="center">0</td>
    <td align="center">0</td>
    <td align="center">0</td>
    <td align="center">0</td>
  </tr>
</table>

### Including White pixels
* Accuracy: 90.68%
* Cohen's Kappa: 0.8776
* Confusion Matrix:

<table align="center">
  <tr>
    <td align="center">3177778</td>
    <td align="center">674</td>
    <td align="center">3042</td>
    <td align="center">1798</td>
    <td align="center">26138</td>
    <td align="center">716</td>
    <td align="center">0</td>
    <td align="center">525</td>
    <td align="center">260984</td>
  </tr>
  <tr>
    <td align="center">725</td>
    <td align="center">1114190</td>
    <td align="center">89</td>
    <td align="center">0</td>
    <td align="center">59</td>
    <td align="center">0</td>
    <td align="center">0</td>
    <td align="center">2</td>
    <td align="center">10788</td>
  </tr>
  <tr>
    <td align="center">2297</td>
    <td align="center">219</td>
    <td align="center">1610602</td>
    <td align="center">6947</td>
    <td align="center">532</td>
    <td align="center">4</td>
    <td align="center">0</td>
    <td align="center">68</td>
    <td align="center">245577</td>
  </tr>
  <tr>
    <td align="center">971</td>
    <td align="center">0</td>
    <td align="center">5887</td>
    <td align="center">1626222</td>
    <td align="center">338</td>
    <td align="center">300</td>
    <td align="center">0</td>
    <td align="center">1</td>
    <td align="center">84661</td>
  </tr>
  <tr>
    <td align="center">25642</td>
    <td align="center">10</td>
    <td align="center">1039</td>
    <td align="center">439</td>
    <td align="center">4823763</td>
    <td align="center">156</td>
    <td align="center">1</td>
    <td align="center">2844</td>
    <td align="center">346078</td>
  </tr>
  <tr>
    <td align="center">659</td>
    <td align="center">0</td>
    <td align="center">124</td>
    <td align="center">1316</td>
    <td align="center">329</td>
    <td align="center">194079</td>
    <td align="center">0</td>
    <td align="center">34</td>
    <td align="center">5565</td>
  </tr>
  <tr>
    <td align="center">0</td>
    <td align="center">0</td>
    <td align="center">0</td>
    <td align="center">34</td>
    <td align="center">83</td>
    <td align="center">0</td>
    <td align="center">26891</td>
    <td align="center">0</td>
    <td align="center">1987</td>
  </tr>
  <tr>
    <td align="center">1220</td>
    <td align="center">0</td>
    <td align="center">41</td>
    <td align="center">0</td>
    <td align="center">1306</td>
    <td align="center">48</td>
    <td align="center">0</td>
    <td align="center">311439</td>
    <td align="center">10363</td>
  </tr>
  <tr>
    <td align="center">335756</td>
    <td align="center">20598</td>
    <td align="center">253679</td>
    <td align="center">102060</td>
    <td align="center">327005</td>
    <td align="center">5708</td>
    <td align="center">2668</td>
    <td align="center">12301</td>
    <td align="center">7678057</td>
  </tr>
</table>

Output images by inference on the test dataset are present in [this directory](./test_results).

## Info
All python scripts are executable and use argparse for commandline arguments. More info about a script's arguments can be obtained by:
```
./script.py -h
```
Docstrings are provided for almost all public functions and classes for further info.
